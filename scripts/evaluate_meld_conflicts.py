#!/usr/bin/env python3
"""Evaluate MLP modality preference on MELD utterance-video conflict pairs."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset

from evaluate_mlp import EmotionMLP, choose_device, compute_metrics, standardize_with_checkpoint

DEFAULT_LABELS = ["anger", "disgust", "fear", "joy", "neutral", "sadness", "surprise"]
PAIR_COLUMNS = {
    "pair_id",
    "split",
    "pair_type",
    "is_conflict",
    "utterance_source_emotion",
    "video_source_emotion",
    "utterance_emotion_id",
    "video_emotion_id",
    "utterance_sample_id",
    "video_sample_id",
    "utterance",
    "video_file",
    "video_path",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a fixed MLP on Qwen embeddings from MELD conflict pairs."
    )
    parser.add_argument("--checkpoint", type=Path, required=True, help="Joint video+utterance MLP checkpoint.")
    parser.add_argument("--embeddings-pt", type=Path, required=True, help="Merged conflict embedding payload.")
    parser.add_argument("--pair-csv", type=Path, required=True, help="MELD conflict matrix CSV.")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"], default="auto")
    parser.add_argument(
        "--require-complete",
        action="store_true",
        help="Fail if any conflict pair is missing an embedding or the payload contains extraction errors.",
    )
    return parser.parse_args()


def load_payload(path: Path) -> dict[str, Any]:
    return torch.load(path, map_location="cpu", weights_only=False)


def load_conflict_pairs(path: Path, label_names: list[str]) -> pd.DataFrame:
    frame = pd.read_csv(path)
    missing = sorted(PAIR_COLUMNS.difference(frame.columns))
    if missing:
        raise ValueError(f"{path} is missing required columns: {missing}")

    frame = frame.loc[
        (frame["split"].astype(str) == "dev")
        & (frame["pair_type"].astype(str) == "cross_class_conflict")
    ].copy()
    if frame.empty:
        raise ValueError(f"No dev cross_class_conflict rows found in {path}")
    if frame["pair_id"].duplicated().any():
        raise ValueError(f"Duplicate pair_id values found in {path}")
    conflict_flags = frame["is_conflict"].astype(str).str.strip().str.lower().isin(
        {"1", "true", "yes"}
    )
    if not conflict_flags.all():
        raise ValueError("Filtered conflict rows contain is_conflict=False")
    if (frame["utterance_source_emotion"] == frame["video_source_emotion"]).any():
        raise ValueError("Conflict rows contain matching utterance and video source labels")

    label_to_id = {name: index for index, name in enumerate(label_names)}
    unknown = sorted(
        set(frame["utterance_source_emotion"])
        .union(frame["video_source_emotion"])
        .difference(label_to_id)
    )
    if unknown:
        raise ValueError(f"Conflict CSV contains labels absent from checkpoint: {unknown}")

    expected_utterance_ids = frame["utterance_source_emotion"].map(label_to_id).astype(int)
    expected_video_ids = frame["video_source_emotion"].map(label_to_id).astype(int)
    if not (frame["utterance_emotion_id"].astype(int) == expected_utterance_ids).all():
        raise ValueError("utterance_emotion_id values do not match checkpoint label ordering")
    if not (frame["video_emotion_id"].astype(int) == expected_video_ids).all():
        raise ValueError("video_emotion_id values do not match checkpoint label ordering")

    return frame.reset_index(drop=True)


def predict_log_probs(
    model: EmotionMLP,
    embeddings: torch.Tensor,
    batch_size: int,
    device: torch.device,
) -> torch.Tensor:
    model.eval()
    outputs = []
    loader = DataLoader(TensorDataset(embeddings), batch_size=batch_size, shuffle=False)
    with torch.inference_mode():
        for (batch_x,) in loader:
            outputs.append(torch.log_softmax(model(batch_x.to(device)), dim=-1).cpu())
    return torch.cat(outputs, dim=0)


def align_embeddings(
    payload: dict[str, Any],
    pairs: pd.DataFrame,
) -> tuple[torch.Tensor, pd.DataFrame, int, list[str], list[str]]:
    embeddings = payload["embeddings"].float()
    sample_ids = [str(item) for item in payload.get("sample_ids", [])]
    labels = payload.get("labels")

    if embeddings.ndim != 2:
        raise ValueError(f"Expected 2D embeddings, got {tuple(embeddings.shape)}")
    if len(sample_ids) != embeddings.shape[0]:
        raise ValueError("Embedding rows and sample_ids lengths differ")
    if len(sample_ids) != len(set(sample_ids)):
        raise ValueError("Embedding payload contains duplicate sample_ids")
    if labels is not None and labels.shape[0] != embeddings.shape[0]:
        raise ValueError("Embedding rows and labels lengths differ")

    pair_index = {str(pair_id): index for index, pair_id in enumerate(pairs["pair_id"])}
    aligned_embeddings = []
    aligned_pair_rows = []
    dropped_non_finite = 0
    extra_embedding_ids = []

    for embedding_index, sample_id in enumerate(sample_ids):
        pair_position = pair_index.get(sample_id)
        if pair_position is None:
            extra_embedding_ids.append(sample_id)
            continue
        embedding = embeddings[embedding_index]
        if not torch.isfinite(embedding).all():
            dropped_non_finite += 1
            continue
        pair = pairs.iloc[pair_position]
        if labels is not None and int(labels[embedding_index]) != int(pair["utterance_emotion_id"]):
            raise ValueError(f"Payload label does not match utterance source label for pair_id={sample_id}")
        aligned_embeddings.append(embedding)
        aligned_pair_rows.append(pair_position)

    if not aligned_embeddings:
        raise ValueError("No finite embeddings aligned with the dev conflict pairs")

    aligned_pairs = pairs.iloc[aligned_pair_rows].reset_index(drop=True)
    aligned_ids = set(aligned_pairs["pair_id"].astype(str))
    missing_pair_ids = [pair_id for pair_id in pairs["pair_id"].astype(str) if pair_id not in aligned_ids]
    return (
        torch.stack(aligned_embeddings),
        aligned_pairs,
        dropped_non_finite,
        missing_pair_ids,
        extra_embedding_ids,
    )


def agreement_metrics(
    source_ids: np.ndarray,
    predictions: np.ndarray,
    label_names: list[str],
) -> dict[str, Any]:
    metrics = compute_metrics(source_ids, predictions, label_names)
    return {
        "agreement_accuracy": metrics["accuracy"],
        "macro_f1": metrics["macro_f1"],
        "weighted_f1": metrics["weighted_f1"],
        "classification_report": metrics["classification_report"],
        "confusion_matrix": metrics["confusion_matrix"],
    }


def make_prediction_rows(
    pairs: pd.DataFrame,
    predictions: np.ndarray,
    probabilities: np.ndarray,
    log_probabilities: np.ndarray,
    label_names: list[str],
) -> list[dict[str, Any]]:
    rows = []
    for index, pair in pairs.iterrows():
        utterance_id = int(pair["utterance_emotion_id"])
        video_id = int(pair["video_emotion_id"])
        prediction_id = int(predictions[index])
        margin = float(log_probabilities[index, utterance_id] - log_probabilities[index, video_id])
        row = {
            "pair_id": pair["pair_id"],
            "utterance_sample_id": pair["utterance_sample_id"],
            "video_sample_id": pair["video_sample_id"],
            "utterance_source_emotion": pair["utterance_source_emotion"],
            "video_source_emotion": pair["video_source_emotion"],
            "pred_id": prediction_id,
            "pred_label": label_names[prediction_id],
            "confidence": float(probabilities[index, prediction_id]),
            "follows_utterance": prediction_id == utterance_id,
            "follows_video": prediction_id == video_id,
            "follows_other": prediction_id not in {utterance_id, video_id},
            "utterance_probability": float(probabilities[index, utterance_id]),
            "video_probability": float(probabilities[index, video_id]),
            "utterance_log_probability": float(log_probabilities[index, utterance_id]),
            "video_log_probability": float(log_probabilities[index, video_id]),
            "utterance_minus_video_log_probability": margin,
            "prefers_utterance_over_video": margin > 0,
            "utterance": pair["utterance"],
            "video_file": pair["video_file"],
            "video_path": pair["video_path"],
        }
        for class_id, class_name in enumerate(label_names):
            row[f"prob_{class_name}"] = float(probabilities[index, class_id])
        rows.append(row)
    return rows


def summarize_pairs(predictions: pd.DataFrame, label_names: list[str]) -> pd.DataFrame:
    rows = []
    grouped = predictions.groupby(
        ["utterance_source_emotion", "video_source_emotion"], sort=False
    )
    for (utterance_emotion, video_emotion), group in grouped:
        row = {
            "utterance_source_emotion": utterance_emotion,
            "video_source_emotion": video_emotion,
            "support": len(group),
            "utterance_follow_rate": group["follows_utterance"].mean(),
            "video_follow_rate": group["follows_video"].mean(),
            "other_prediction_rate": group["follows_other"].mean(),
            "utterance_preference_rate": group["prefers_utterance_over_video"].mean(),
            "mean_utterance_probability": group["utterance_probability"].mean(),
            "mean_video_probability": group["video_probability"].mean(),
            "mean_utterance_minus_video_log_probability": group[
                "utterance_minus_video_log_probability"
            ].mean(),
        }
        prediction_counts = group["pred_label"].value_counts()
        for class_name in label_names:
            row[f"predicted_{class_name}"] = int(prediction_counts.get(class_name, 0))
        rows.append(row)
    return pd.DataFrame(rows)


def write_matrix(
    pair_summary: pd.DataFrame,
    value_column: str,
    label_names: list[str],
    path: Path,
) -> None:
    matrix = pair_summary.pivot(
        index="utterance_source_emotion",
        columns="video_source_emotion",
        values=value_column,
    ).reindex(index=label_names, columns=label_names)
    matrix.to_csv(path)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = choose_device(args.device)
    checkpoint = load_payload(args.checkpoint)
    payload = load_payload(args.embeddings_pt)
    label_names = [str(item) for item in checkpoint.get("label_names", DEFAULT_LABELS)]
    pairs = load_conflict_pairs(args.pair_csv, label_names)
    num_conflict_pairs = len(pairs)
    embeddings, pairs, dropped_non_finite, missing_pair_ids, extra_embedding_ids = align_embeddings(
        payload, pairs
    )

    errors = list(payload.get("errors", []))
    if args.require_complete and (missing_pair_ids or errors or dropped_non_finite):
        raise ValueError(
            "Incomplete conflict evaluation: "
            f"missing_pairs={len(missing_pair_ids)}, extraction_errors={len(errors)}, "
            f"non_finite={dropped_non_finite}"
        )

    input_dim = int(checkpoint["input_dim"])
    if embeddings.shape[1] != input_dim:
        raise ValueError(
            f"Embedding dim {embeddings.shape[1]} does not match checkpoint input_dim {input_dim}"
        )
    standardized = standardize_with_checkpoint(embeddings, checkpoint)
    model = EmotionMLP(
        input_dim=input_dim,
        hidden_dim=int(checkpoint["hidden_dim"]),
        num_classes=int(checkpoint.get("num_classes", len(label_names))),
        dropout=float(checkpoint.get("dropout", 0.0)),
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])

    log_probabilities_tensor = predict_log_probs(model, standardized, args.batch_size, device)
    log_probabilities = log_probabilities_tensor.numpy()
    probabilities = log_probabilities_tensor.exp().numpy()
    predictions = probabilities.argmax(axis=1)
    utterance_ids = pairs["utterance_emotion_id"].to_numpy(dtype=np.int64)
    video_ids = pairs["video_emotion_id"].to_numpy(dtype=np.int64)

    prediction_rows = make_prediction_rows(
        pairs, predictions, probabilities, log_probabilities, label_names
    )
    prediction_frame = pd.DataFrame(prediction_rows)
    pair_summary = summarize_pairs(prediction_frame, label_names)

    predictions_path = args.output_dir / "dev_conflict_predictions.csv"
    pair_summary_path = args.output_dir / "dev_conflict_pair_summary.csv"
    prediction_frame.to_csv(predictions_path, index=False, quoting=csv.QUOTE_MINIMAL)
    pair_summary.to_csv(pair_summary_path, index=False)
    write_matrix(
        pair_summary,
        "utterance_follow_rate",
        label_names,
        args.output_dir / "dev_utterance_follow_rate_matrix.csv",
    )
    write_matrix(
        pair_summary,
        "video_follow_rate",
        label_names,
        args.output_dir / "dev_video_follow_rate_matrix.csv",
    )
    write_matrix(
        pair_summary,
        "mean_utterance_minus_video_log_probability",
        label_names,
        args.output_dir / "dev_utterance_video_log_probability_margin_matrix.csv",
    )

    utterance_follow = prediction_frame["follows_utterance"].mean()
    video_follow = prediction_frame["follows_video"].mean()
    other_rate = prediction_frame["follows_other"].mean()
    summary = {
        "dataset": "MELD dev cross-class utterance-video conflicts",
        "interpretation_note": (
            "MELD source emotions are inherited multimodal labels, not independently verified "
            "utterance-only or video-only ground truth. Agreement metrics quantify model preference."
        ),
        "num_conflict_pairs_in_csv": int(num_conflict_pairs),
        "num_evaluated_pairs": int(len(prediction_frame)),
        "num_missing_pair_embeddings": len(missing_pair_ids),
        "num_extra_embedding_ids": len(extra_embedding_ids),
        "num_non_finite_embeddings_dropped": dropped_non_finite,
        "num_extraction_errors": len(errors),
        "overall": {
            "utterance_follow_rate": float(utterance_follow),
            "video_follow_rate": float(video_follow),
            "other_prediction_rate": float(other_rate),
            "utterance_preference_rate": float(
                prediction_frame["prefers_utterance_over_video"].mean()
            ),
            "mean_utterance_probability": float(prediction_frame["utterance_probability"].mean()),
            "mean_video_probability": float(prediction_frame["video_probability"].mean()),
            "mean_utterance_minus_video_log_probability": float(
                prediction_frame["utterance_minus_video_log_probability"].mean()
            ),
        },
        "prediction_vs_utterance_source_label": agreement_metrics(
            utterance_ids, predictions, label_names
        ),
        "prediction_vs_video_source_label": agreement_metrics(video_ids, predictions, label_names),
        "files": {
            "predictions": str(predictions_path),
            "ordered_pair_summary": str(pair_summary_path),
        },
    }
    summary_path = args.output_dir / "dev_conflict_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    print(f"Device: {device}")
    print(f"Evaluated conflict pairs: {len(prediction_frame):,}/{summary['num_conflict_pairs_in_csv']:,}")
    print(f"Utterance follow rate: {utterance_follow:.4f}")
    print(f"Video follow rate: {video_follow:.4f}")
    print(f"Other prediction rate: {other_rate:.4f}")
    print(
        "Mean utterance - video log-probability: "
        f"{summary['overall']['mean_utterance_minus_video_log_probability']:.4f}"
    )
    print(f"Saved summary: {summary_path}")
    print(f"Saved predictions: {predictions_path}")


if __name__ == "__main__":
    main()
