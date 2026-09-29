#!/usr/bin/env python3
"""Compare full and zero-visual-token Qwen embeddings with one fixed MLP."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from evaluate_mlp import EmotionMLP, choose_device, compute_metrics, standardize_with_checkpoint

DIRECT_ZERO_IMPLEMENTATION = "direct_placeholder_zero"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Paired evaluation of full and zero-visual-token embeddings.")
    parser.add_argument("--checkpoint", type=Path, required=True, help="Joint video+utterance MLP checkpoint.")
    parser.add_argument("--full-pt", type=Path, required=True, help="Existing full video+utterance embeddings.")
    parser.add_argument("--zero-video-pt", type=Path, required=True, help="Embeddings extracted with zero visual tokens.")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--split-name", default="test")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"], default="auto")
    return parser.parse_args()


def load_payload(path: Path) -> dict[str, Any]:
    return torch.load(path, map_location="cpu", weights_only=False)


def visual_token_ablation_modes(payload: dict[str, Any]) -> set[str]:
    config = payload.get("config", {})
    if "visual_token_ablation" in config:
        return {str(config["visual_token_ablation"])}
    return {
        str(item.get("config", {}).get("visual_token_ablation", "none"))
        for item in config.get("merged_from", [])
        if isinstance(item, dict)
    }


def visual_ablation_implementations(payload: dict[str, Any]) -> set[str]:
    config = payload.get("config", {})
    if "visual_ablation_implementation" in config:
        return {str(config["visual_ablation_implementation"])}
    return {
        str(item.get("config", {}).get("visual_ablation_implementation"))
        for item in config.get("merged_from", [])
        if isinstance(item, dict) and item.get("config", {}).get("visual_ablation_implementation") is not None
    }


def validate_unique_ids(sample_ids: list[str], path: Path) -> None:
    if len(sample_ids) != len(set(sample_ids)):
        raise ValueError(f"Duplicate sample_ids found in {path}")


def align_payloads(
    full_payload: dict[str, Any],
    zero_payload: dict[str, Any],
    full_path: Path,
    zero_path: Path,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, list[str], list[dict[str, Any]], int]:
    full_x = full_payload["embeddings"].float()
    zero_x = zero_payload["embeddings"].float()
    full_y = full_payload["labels"].long()
    zero_y = zero_payload["labels"].long()
    full_ids = [str(item) for item in full_payload.get("sample_ids", [])]
    zero_ids = [str(item) for item in zero_payload.get("sample_ids", [])]

    if full_x.ndim != 2 or zero_x.ndim != 2:
        raise ValueError(f"Expected 2D embeddings, got {tuple(full_x.shape)} and {tuple(zero_x.shape)}")
    if full_x.shape[1] != zero_x.shape[1]:
        raise ValueError(f"Embedding dimensions differ: {full_x.shape[1]} vs {zero_x.shape[1]}")
    if len(full_ids) != full_x.shape[0] or len(zero_ids) != zero_x.shape[0]:
        raise ValueError("Embedding rows and sample_ids lengths differ")
    if full_y.shape[0] != full_x.shape[0] or zero_y.shape[0] != zero_x.shape[0]:
        raise ValueError("Embedding rows and labels lengths differ")

    validate_unique_ids(full_ids, full_path)
    validate_unique_ids(zero_ids, zero_path)
    zero_index = {sample_id: i for i, sample_id in enumerate(zero_ids)}
    full_metadata = full_payload.get("metadata", [{} for _ in full_ids])

    full_rows = []
    zero_rows = []
    labels = []
    sample_ids = []
    metadata = []
    dropped_non_finite = 0

    for full_i, sample_id in enumerate(full_ids):
        zero_i = zero_index.get(sample_id)
        if zero_i is None:
            continue
        if int(full_y[full_i]) != int(zero_y[zero_i]):
            raise ValueError(f"Label mismatch for sample_id={sample_id}")
        if not torch.isfinite(full_x[full_i]).all() or not torch.isfinite(zero_x[zero_i]).all():
            dropped_non_finite += 1
            continue

        full_rows.append(full_x[full_i])
        zero_rows.append(zero_x[zero_i])
        labels.append(full_y[full_i])
        sample_ids.append(sample_id)
        metadata.append(full_metadata[full_i] if full_i < len(full_metadata) else {})

    if not sample_ids:
        raise ValueError("No aligned finite samples were found")

    return (
        torch.stack(full_rows),
        torch.stack(zero_rows),
        torch.stack(labels).long(),
        sample_ids,
        metadata,
        dropped_non_finite,
    )


def predict_log_probs(
    model: EmotionMLP,
    x: torch.Tensor,
    batch_size: int,
    device: torch.device,
) -> torch.Tensor:
    model.eval()
    outputs = []
    loader = DataLoader(TensorDataset(x), batch_size=batch_size, shuffle=False)
    with torch.inference_mode():
        for (batch_x,) in loader:
            logits = model(batch_x.to(device))
            outputs.append(torch.log_softmax(logits, dim=-1).cpu())
    return torch.cat(outputs, dim=0)


def condition_metrics(y_true: np.ndarray, log_probs: torch.Tensor, label_names: list[str]) -> dict[str, Any]:
    predictions = log_probs.argmax(dim=-1).numpy()
    return compute_metrics(y_true, predictions, label_names)


def summarize_deltas(
    labels: torch.Tensor,
    full_log_probs: torch.Tensor,
    zero_log_probs: torch.Tensor,
    label_names: list[str],
) -> dict[str, Any]:
    row_index = torch.arange(labels.shape[0])
    full_gold = full_log_probs[row_index, labels]
    zero_gold = zero_log_probs[row_index, labels]
    gold_delta = full_gold - zero_gold
    full_pred = full_log_probs.argmax(dim=-1)
    zero_pred = zero_log_probs.argmax(dim=-1)
    full_correct = full_pred.eq(labels)
    zero_correct = zero_pred.eq(labels)

    per_true_class = {}
    for class_id, class_name in enumerate(label_names):
        mask = labels.eq(class_id)
        class_delta = gold_delta[mask]
        per_true_class[class_name] = {
            "support": int(mask.sum()),
            "mean_full_gold_log_probability": float(full_gold[mask].mean()),
            "mean_zero_video_gold_log_probability": float(zero_gold[mask].mean()),
            "mean_visual_delta_gold_log_probability": float(class_delta.mean()),
            "visual_benefit_rate": float(class_delta.gt(0).float().mean()),
        }

    per_output_class = {}
    all_class_deltas = full_log_probs - zero_log_probs
    for class_id, class_name in enumerate(label_names):
        per_output_class[class_name] = {
            "mean_full_log_probability": float(full_log_probs[:, class_id].mean()),
            "mean_zero_video_log_probability": float(zero_log_probs[:, class_id].mean()),
            "mean_visual_delta_log_probability": float(all_class_deltas[:, class_id].mean()),
        }

    return {
        "mean_full_gold_log_probability": float(full_gold.mean()),
        "mean_zero_video_gold_log_probability": float(zero_gold.mean()),
        "mean_visual_delta_gold_log_probability": float(gold_delta.mean()),
        "visual_benefit_rate": float(gold_delta.gt(0).float().mean()),
        "prediction_change_rate": float(full_pred.ne(zero_pred).float().mean()),
        "zero_wrong_to_full_correct": int((~zero_correct & full_correct).sum()),
        "zero_correct_to_full_wrong": int((zero_correct & ~full_correct).sum()),
        "per_true_class": per_true_class,
        "per_output_class": per_output_class,
    }


def write_paired_predictions(
    path: Path,
    sample_ids: list[str],
    metadata: list[dict[str, Any]],
    labels: torch.Tensor,
    full_log_probs: torch.Tensor,
    zero_log_probs: torch.Tensor,
    label_names: list[str],
) -> None:
    full_pred = full_log_probs.argmax(dim=-1)
    zero_pred = zero_log_probs.argmax(dim=-1)
    fields = [
        "sample_id",
        "true_id",
        "true_label",
        "utterance",
        "video_file",
        "full_pred_id",
        "full_pred_label",
        "zero_video_pred_id",
        "zero_video_pred_label",
        "full_correct",
        "zero_video_correct",
        "full_gold_log_probability",
        "zero_video_gold_log_probability",
        "visual_delta_gold_log_probability",
        "full_gold_probability",
        "zero_video_gold_probability",
    ]
    for class_name in label_names:
        fields.extend(
            [
                f"full_prob_{class_name}",
                f"zero_video_prob_{class_name}",
                f"full_log_prob_{class_name}",
                f"zero_video_log_prob_{class_name}",
                f"visual_delta_log_prob_{class_name}",
            ]
        )

    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for i, sample_id in enumerate(sample_ids):
            true_id = int(labels[i])
            full_pred_id = int(full_pred[i])
            zero_pred_id = int(zero_pred[i])
            meta = metadata[i] if isinstance(metadata[i], dict) else {}
            row = {
                "sample_id": sample_id,
                "true_id": true_id,
                "true_label": label_names[true_id],
                "utterance": meta.get("utterance", ""),
                "video_file": meta.get("video_file", meta.get("video_path", "")),
                "full_pred_id": full_pred_id,
                "full_pred_label": label_names[full_pred_id],
                "zero_video_pred_id": zero_pred_id,
                "zero_video_pred_label": label_names[zero_pred_id],
                "full_correct": full_pred_id == true_id,
                "zero_video_correct": zero_pred_id == true_id,
                "full_gold_log_probability": float(full_log_probs[i, true_id]),
                "zero_video_gold_log_probability": float(zero_log_probs[i, true_id]),
                "visual_delta_gold_log_probability": float(
                    full_log_probs[i, true_id] - zero_log_probs[i, true_id]
                ),
                "full_gold_probability": float(full_log_probs[i, true_id].exp()),
                "zero_video_gold_probability": float(zero_log_probs[i, true_id].exp()),
            }
            for class_id, class_name in enumerate(label_names):
                row[f"full_prob_{class_name}"] = float(full_log_probs[i, class_id].exp())
                row[f"zero_video_prob_{class_name}"] = float(zero_log_probs[i, class_id].exp())
                row[f"full_log_prob_{class_name}"] = float(full_log_probs[i, class_id])
                row[f"zero_video_log_prob_{class_name}"] = float(zero_log_probs[i, class_id])
                row[f"visual_delta_log_prob_{class_name}"] = float(
                    full_log_probs[i, class_id] - zero_log_probs[i, class_id]
                )
            writer.writerow(row)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = choose_device(args.device)

    checkpoint = load_payload(args.checkpoint)
    full_payload = load_payload(args.full_pt)
    zero_payload = load_payload(args.zero_video_pt)
    full_modes = visual_token_ablation_modes(full_payload)
    if full_modes and full_modes != {"none"}:
        raise ValueError(
            f"Expected --full-pt to contain only normal embeddings, found modes: {sorted(full_modes)}"
        )
    zero_modes = visual_token_ablation_modes(zero_payload)
    if zero_modes and zero_modes != {"zero"}:
        raise ValueError(
            f"Expected --zero-video-pt to contain only zero-token embeddings, found modes: {sorted(zero_modes)}"
        )
    zero_implementations = visual_ablation_implementations(zero_payload)
    if zero_implementations != {DIRECT_ZERO_IMPLEMENTATION}:
        raise ValueError(
            "Expected direct placeholder-zero embeddings, found implementations: "
            f"{sorted(zero_implementations)}"
        )
    label_names = checkpoint.get("label_names", full_payload.get("label_names"))
    if not label_names:
        raise ValueError("No label_names found in checkpoint or embedding payload")

    full_x, zero_x, labels, sample_ids, metadata, dropped_non_finite = align_payloads(
        full_payload,
        zero_payload,
        args.full_pt,
        args.zero_video_pt,
    )
    input_dim = int(checkpoint["input_dim"])
    if full_x.shape[1] != input_dim:
        raise ValueError(f"Embedding dim {full_x.shape[1]} does not match checkpoint input_dim {input_dim}")

    model = EmotionMLP(
        input_dim=input_dim,
        hidden_dim=int(checkpoint["hidden_dim"]),
        num_classes=int(checkpoint.get("num_classes", len(label_names))),
        dropout=float(checkpoint.get("dropout", 0.0)),
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])

    full_x = standardize_with_checkpoint(full_x, checkpoint)
    zero_x = standardize_with_checkpoint(zero_x, checkpoint)
    full_log_probs = predict_log_probs(model, full_x, args.batch_size, device)
    zero_log_probs = predict_log_probs(model, zero_x, args.batch_size, device)
    y_true = labels.numpy()

    full_metrics = condition_metrics(y_true, full_log_probs, label_names)
    zero_metrics = condition_metrics(y_true, zero_log_probs, label_names)
    summary = {
        "split": args.split_name,
        "num_aligned_samples": len(sample_ids),
        "dropped_non_finite_pairs": dropped_non_finite,
        "full_metrics": full_metrics,
        "zero_video_metrics": zero_metrics,
        "metric_deltas_full_minus_zero": {
            "accuracy": full_metrics["accuracy"] - zero_metrics["accuracy"],
            "macro_f1": full_metrics["macro_f1"] - zero_metrics["macro_f1"],
            "weighted_f1": full_metrics["weighted_f1"] - zero_metrics["weighted_f1"],
        },
        "log_probability_analysis": summarize_deltas(labels, full_log_probs, zero_log_probs, label_names),
        "checkpoint": str(args.checkpoint),
        "full_embeddings": str(args.full_pt),
        "zero_video_embeddings": str(args.zero_video_pt),
    }

    summary_path = args.output_dir / f"{args.split_name}_visual_ablation_summary.json"
    predictions_path = args.output_dir / f"{args.split_name}_visual_ablation_predictions.csv"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    write_paired_predictions(
        predictions_path,
        sample_ids,
        metadata,
        labels,
        full_log_probs,
        zero_log_probs,
        label_names,
    )

    print(f"Device: {device}")
    print(f"Aligned samples: {len(sample_ids)}")
    print(f"Full accuracy / macro F1: {full_metrics['accuracy']:.4f} / {full_metrics['macro_f1']:.4f}")
    print(f"Zero-video accuracy / macro F1: {zero_metrics['accuracy']:.4f} / {zero_metrics['macro_f1']:.4f}")
    print(
        "Mean visual delta for gold-class log probability: "
        f"{summary['log_probability_analysis']['mean_visual_delta_gold_log_probability']:.6f}"
    )
    print(f"Saved summary: {summary_path}")
    print(f"Saved paired predictions: {predictions_path}")


if __name__ == "__main__":
    main()
