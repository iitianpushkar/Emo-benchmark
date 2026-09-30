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
DIRECT_UTTERANCE_ZERO_IMPLEMENTATION = "direct_utterance_zero"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Paired evaluation of full and zeroed-modality-token embeddings.")
    parser.add_argument("--checkpoint", type=Path, required=True, help="Joint video+utterance MLP checkpoint.")
    parser.add_argument("--full-pt", type=Path, required=True, help="Existing full video+utterance embeddings.")
    ablation_group = parser.add_mutually_exclusive_group(required=True)
    ablation_group.add_argument("--zero-video-pt", type=Path, help="Embeddings extracted with zero visual tokens.")
    ablation_group.add_argument(
        "--zero-utterance-pt",
        type=Path,
        help="Embeddings extracted with zero utterance tokens.",
    )
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


def utterance_token_ablation_modes(payload: dict[str, Any]) -> set[str]:
    config = payload.get("config", {})
    if "utterance_token_ablation" in config:
        return {str(config["utterance_token_ablation"])}
    return {
        str(item.get("config", {}).get("utterance_token_ablation", "none"))
        for item in config.get("merged_from", [])
        if isinstance(item, dict)
    }


def utterance_ablation_implementations(payload: dict[str, Any]) -> set[str]:
    config = payload.get("config", {})
    if "utterance_ablation_implementation" in config:
        return {str(config["utterance_ablation_implementation"])}
    return {
        str(item.get("config", {}).get("utterance_ablation_implementation"))
        for item in config.get("merged_from", [])
        if isinstance(item, dict) and item.get("config", {}).get("utterance_ablation_implementation") is not None
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
    ablated_log_probs: torch.Tensor,
    label_names: list[str],
    ablated_key: str,
    effect_key: str,
) -> dict[str, Any]:
    row_index = torch.arange(labels.shape[0])
    full_gold = full_log_probs[row_index, labels]
    ablated_gold = ablated_log_probs[row_index, labels]
    gold_delta = full_gold - ablated_gold
    full_pred = full_log_probs.argmax(dim=-1)
    ablated_pred = ablated_log_probs.argmax(dim=-1)
    full_correct = full_pred.eq(labels)
    ablated_correct = ablated_pred.eq(labels)

    per_true_class = {}
    for class_id, class_name in enumerate(label_names):
        mask = labels.eq(class_id)
        class_delta = gold_delta[mask]
        per_true_class[class_name] = {
            "support": int(mask.sum()),
            "mean_full_gold_log_probability": float(full_gold[mask].mean()),
            f"mean_{ablated_key}_gold_log_probability": float(ablated_gold[mask].mean()),
            f"mean_{effect_key}_delta_gold_log_probability": float(class_delta.mean()),
            f"{effect_key}_benefit_rate": float(class_delta.gt(0).float().mean()),
        }

    per_output_class = {}
    all_class_deltas = full_log_probs - ablated_log_probs
    for class_id, class_name in enumerate(label_names):
        per_output_class[class_name] = {
            "mean_full_log_probability": float(full_log_probs[:, class_id].mean()),
            f"mean_{ablated_key}_log_probability": float(ablated_log_probs[:, class_id].mean()),
            f"mean_{effect_key}_delta_log_probability": float(all_class_deltas[:, class_id].mean()),
        }

    return {
        "mean_full_gold_log_probability": float(full_gold.mean()),
        f"mean_{ablated_key}_gold_log_probability": float(ablated_gold.mean()),
        f"mean_{effect_key}_delta_gold_log_probability": float(gold_delta.mean()),
        f"{effect_key}_benefit_rate": float(gold_delta.gt(0).float().mean()),
        "prediction_change_rate": float(full_pred.ne(ablated_pred).float().mean()),
        "ablated_wrong_to_full_correct": int((~ablated_correct & full_correct).sum()),
        "ablated_correct_to_full_wrong": int((ablated_correct & ~full_correct).sum()),
        "per_true_class": per_true_class,
        "per_output_class": per_output_class,
    }


def write_paired_predictions(
    path: Path,
    sample_ids: list[str],
    metadata: list[dict[str, Any]],
    labels: torch.Tensor,
    full_log_probs: torch.Tensor,
    ablated_log_probs: torch.Tensor,
    label_names: list[str],
    ablated_key: str,
    effect_key: str,
) -> None:
    full_pred = full_log_probs.argmax(dim=-1)
    ablated_pred = ablated_log_probs.argmax(dim=-1)
    fields = [
        "sample_id",
        "true_id",
        "true_label",
        "utterance",
        "video_file",
        "full_pred_id",
        "full_pred_label",
        f"{ablated_key}_pred_id",
        f"{ablated_key}_pred_label",
        "full_correct",
        f"{ablated_key}_correct",
        "full_gold_log_probability",
        f"{ablated_key}_gold_log_probability",
        f"{effect_key}_delta_gold_log_probability",
        "full_gold_probability",
        f"{ablated_key}_gold_probability",
    ]
    for class_name in label_names:
        fields.extend(
            [
                f"full_prob_{class_name}",
                f"{ablated_key}_prob_{class_name}",
                f"full_log_prob_{class_name}",
                f"{ablated_key}_log_prob_{class_name}",
                f"{effect_key}_delta_log_prob_{class_name}",
            ]
        )

    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for i, sample_id in enumerate(sample_ids):
            true_id = int(labels[i])
            full_pred_id = int(full_pred[i])
            ablated_pred_id = int(ablated_pred[i])
            meta = metadata[i] if isinstance(metadata[i], dict) else {}
            row = {
                "sample_id": sample_id,
                "true_id": true_id,
                "true_label": label_names[true_id],
                "utterance": meta.get("utterance", ""),
                "video_file": meta.get("video_file", meta.get("video_path", "")),
                "full_pred_id": full_pred_id,
                "full_pred_label": label_names[full_pred_id],
                f"{ablated_key}_pred_id": ablated_pred_id,
                f"{ablated_key}_pred_label": label_names[ablated_pred_id],
                "full_correct": full_pred_id == true_id,
                f"{ablated_key}_correct": ablated_pred_id == true_id,
                "full_gold_log_probability": float(full_log_probs[i, true_id]),
                f"{ablated_key}_gold_log_probability": float(ablated_log_probs[i, true_id]),
                f"{effect_key}_delta_gold_log_probability": float(
                    full_log_probs[i, true_id] - ablated_log_probs[i, true_id]
                ),
                "full_gold_probability": float(full_log_probs[i, true_id].exp()),
                f"{ablated_key}_gold_probability": float(ablated_log_probs[i, true_id].exp()),
            }
            for class_id, class_name in enumerate(label_names):
                row[f"full_prob_{class_name}"] = float(full_log_probs[i, class_id].exp())
                row[f"{ablated_key}_prob_{class_name}"] = float(ablated_log_probs[i, class_id].exp())
                row[f"full_log_prob_{class_name}"] = float(full_log_probs[i, class_id])
                row[f"{ablated_key}_log_prob_{class_name}"] = float(ablated_log_probs[i, class_id])
                row[f"{effect_key}_delta_log_prob_{class_name}"] = float(
                    full_log_probs[i, class_id] - ablated_log_probs[i, class_id]
                )
            writer.writerow(row)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = choose_device(args.device)

    checkpoint = load_payload(args.checkpoint)
    full_payload = load_payload(args.full_pt)
    ablated_path = args.zero_video_pt or args.zero_utterance_pt
    if ablated_path is None:
        raise ValueError("One ablated embedding path is required")
    ablated_payload = load_payload(ablated_path)

    full_visual_modes = visual_token_ablation_modes(full_payload)
    full_utterance_modes = utterance_token_ablation_modes(full_payload)
    if full_visual_modes and full_visual_modes != {"none"}:
        raise ValueError(
            f"Expected --full-pt to contain normal visual tokens, found modes: {sorted(full_visual_modes)}"
        )
    if full_utterance_modes and full_utterance_modes != {"none"}:
        raise ValueError(
            "Expected --full-pt to contain normal utterance tokens, "
            f"found modes: {sorted(full_utterance_modes)}"
        )

    if args.zero_video_pt is not None:
        ablation_slug = "visual"
        ablated_key = "zero_video"
        effect_key = "visual"
        visual_modes = visual_token_ablation_modes(ablated_payload)
        utterance_modes = utterance_token_ablation_modes(ablated_payload)
        if visual_modes != {"zero"} or (utterance_modes and utterance_modes != {"none"}):
            raise ValueError("--zero-video-pt must contain only the video-zero condition")
        implementations = visual_ablation_implementations(ablated_payload)
        if implementations != {DIRECT_ZERO_IMPLEMENTATION}:
            raise ValueError(
                "Expected direct placeholder-zero embeddings, found implementations: "
                f"{sorted(implementations)}"
            )
    else:
        ablation_slug = "utterance"
        ablated_key = "zero_utterance"
        effect_key = "utterance"
        visual_modes = visual_token_ablation_modes(ablated_payload)
        utterance_modes = utterance_token_ablation_modes(ablated_payload)
        if utterance_modes != {"zero"} or (visual_modes and visual_modes != {"none"}):
            raise ValueError("--zero-utterance-pt must contain only the utterance-zero condition")
        implementations = utterance_ablation_implementations(ablated_payload)
        if implementations != {DIRECT_UTTERANCE_ZERO_IMPLEMENTATION}:
            raise ValueError(
                "Expected direct utterance-zero embeddings, found implementations: "
                f"{sorted(implementations)}"
            )
    label_names = checkpoint.get("label_names", full_payload.get("label_names"))
    if not label_names:
        raise ValueError("No label_names found in checkpoint or embedding payload")

    full_x, ablated_x, labels, sample_ids, metadata, dropped_non_finite = align_payloads(
        full_payload,
        ablated_payload,
        args.full_pt,
        ablated_path,
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
    ablated_x = standardize_with_checkpoint(ablated_x, checkpoint)
    full_log_probs = predict_log_probs(model, full_x, args.batch_size, device)
    ablated_log_probs = predict_log_probs(model, ablated_x, args.batch_size, device)
    y_true = labels.numpy()

    full_metrics = condition_metrics(y_true, full_log_probs, label_names)
    ablated_metrics = condition_metrics(y_true, ablated_log_probs, label_names)
    summary = {
        "split": args.split_name,
        "num_aligned_samples": len(sample_ids),
        "dropped_non_finite_pairs": dropped_non_finite,
        "full_metrics": full_metrics,
        f"{ablated_key}_metrics": ablated_metrics,
        "metric_deltas_full_minus_zero": {
            "accuracy": full_metrics["accuracy"] - ablated_metrics["accuracy"],
            "macro_f1": full_metrics["macro_f1"] - ablated_metrics["macro_f1"],
            "weighted_f1": full_metrics["weighted_f1"] - ablated_metrics["weighted_f1"],
        },
        "log_probability_analysis": summarize_deltas(
            labels,
            full_log_probs,
            ablated_log_probs,
            label_names,
            ablated_key,
            effect_key,
        ),
        "checkpoint": str(args.checkpoint),
        "full_embeddings": str(args.full_pt),
        f"{ablated_key}_embeddings": str(ablated_path),
    }

    summary_path = args.output_dir / f"{args.split_name}_{ablation_slug}_ablation_summary.json"
    predictions_path = args.output_dir / f"{args.split_name}_{ablation_slug}_ablation_predictions.csv"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    write_paired_predictions(
        predictions_path,
        sample_ids,
        metadata,
        labels,
        full_log_probs,
        ablated_log_probs,
        label_names,
        ablated_key,
        effect_key,
    )

    print(f"Device: {device}")
    print(f"Aligned samples: {len(sample_ids)}")
    print(f"Full accuracy / macro F1: {full_metrics['accuracy']:.4f} / {full_metrics['macro_f1']:.4f}")
    print(
        f"{ablated_key} accuracy / macro F1: "
        f"{ablated_metrics['accuracy']:.4f} / {ablated_metrics['macro_f1']:.4f}"
    )
    print(
        f"Mean {effect_key} delta for gold-class log probability: "
        f"{summary['log_probability_analysis'][f'mean_{effect_key}_delta_gold_log_probability']:.6f}"
    )
    print(f"Saved summary: {summary_path}")
    print(f"Saved paired predictions: {predictions_path}")


if __name__ == "__main__":
    main()
