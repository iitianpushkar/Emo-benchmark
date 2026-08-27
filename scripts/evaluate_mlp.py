#!/usr/bin/env python3
"""Evaluate a saved MLP checkpoint on extracted VLM embeddings."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

DEFAULT_LABELS = ["anger", "disgust", "fear", "joy", "neutral", "sadness", "surprise"]


class EmotionMLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, num_classes: int, dropout: float) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate saved MLP on embedding file.")
    parser.add_argument("--checkpoint", type=Path, required=True, help="Path to best_mlp.pt.")
    parser.add_argument("--embeddings-pt", type=Path, required=True, help="Embeddings .pt file to evaluate.")
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory for metrics and predictions.")
    parser.add_argument("--split-name", default="eval", help="Name used in output filenames.")
    parser.add_argument("--batch-size", type=int, default=128, help="Evaluation batch size.")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"], default="auto")
    return parser.parse_args()


def choose_device(device_arg: str) -> torch.device:
    if device_arg != "auto":
        return torch.device(device_arg)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_payload(path: Path) -> dict[str, Any]:
    return torch.load(path, map_location="cpu", weights_only=False)


def standardize_with_checkpoint(x: torch.Tensor, checkpoint: dict[str, Any]) -> torch.Tensor:
    mean = checkpoint["feature_mean"].float().view(1, -1)
    std = checkpoint["feature_std"].float().view(1, -1).clamp_min(1e-6)
    return (x.float() - mean) / std


def predict(model: nn.Module, x: torch.Tensor, batch_size: int, device: torch.device) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    logits_all = []
    loader = DataLoader(TensorDataset(x), batch_size=batch_size, shuffle=False)

    with torch.inference_mode():
        for (batch_x,) in loader:
            logits = model(batch_x.to(device))
            logits_all.append(logits.cpu())

    logits = torch.cat(logits_all, dim=0)
    probs = torch.softmax(logits, dim=-1).numpy()
    preds = probs.argmax(axis=1)
    return preds, probs


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, label_names: list[str]) -> dict[str, Any]:
    labels = list(range(len(label_names)))
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", labels=labels, zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", labels=labels, zero_division=0)),
        "classification_report": classification_report(
            y_true,
            y_pred,
            labels=labels,
            target_names=label_names,
            zero_division=0,
            output_dict=True,
        ),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
    }


def write_predictions(
    path: Path,
    payload: dict[str, Any],
    y_true: np.ndarray,
    y_pred: np.ndarray,
    probs: np.ndarray,
    label_names: list[str],
) -> None:
    sample_ids = payload.get("sample_ids", [str(i) for i in range(len(y_true))])
    metadata = payload.get("metadata", [{} for _ in range(len(y_true))])

    fields = [
        "sample_id",
        "true_id",
        "true_label",
        "pred_id",
        "pred_label",
        "correct",
        "confidence",
        "video_file",
        "utterance",
    ]
    fields += [f"prob_{name}" for name in label_names]

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for i, sample_id in enumerate(sample_ids):
            meta = metadata[i] if i < len(metadata) and isinstance(metadata[i], dict) else {}
            row = {
                "sample_id": sample_id,
                "true_id": int(y_true[i]),
                "true_label": label_names[int(y_true[i])],
                "pred_id": int(y_pred[i]),
                "pred_label": label_names[int(y_pred[i])],
                "correct": bool(y_true[i] == y_pred[i]),
                "confidence": float(probs[i, int(y_pred[i])]),
                "video_file": meta.get("video_file", ""),
                "utterance": meta.get("utterance", ""),
            }
            for j, name in enumerate(label_names):
                row[f"prob_{name}"] = float(probs[i, j])
            writer.writerow(row)


def main() -> None:
    args = parse_args()
    device = choose_device(args.device)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    checkpoint = load_payload(args.checkpoint)
    payload = load_payload(args.embeddings_pt)

    label_names = checkpoint.get("label_names", payload.get("label_names", DEFAULT_LABELS))
    input_dim = int(checkpoint["input_dim"])
    hidden_dim = int(checkpoint["hidden_dim"])
    dropout = float(checkpoint.get("dropout", 0.0))
    num_classes = int(checkpoint.get("num_classes", len(label_names)))

    x = payload["embeddings"].float()
    y = payload["labels"].long()

    if x.ndim != 2:
        raise ValueError(f"Expected embeddings shape [num_samples, dim], got {tuple(x.shape)}")
    if x.shape[1] != input_dim:
        raise ValueError(f"Embedding dim {x.shape[1]} does not match checkpoint input_dim {input_dim}")

    x = standardize_with_checkpoint(x, checkpoint)

    model = EmotionMLP(input_dim=input_dim, hidden_dim=hidden_dim, num_classes=num_classes, dropout=dropout).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])

    y_pred, probs = predict(model, x, args.batch_size, device)
    y_true = y.numpy()
    metrics = compute_metrics(y_true, y_pred, label_names)

    metrics_path = args.output_dir / f"{args.split_name}_metrics.json"
    predictions_path = args.output_dir / f"{args.split_name}_predictions.csv"

    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    write_predictions(predictions_path, payload, y_true, y_pred, probs, label_names)

    print(f"Device: {device}")
    print(f"Embeddings: {tuple(x.shape)}")
    print(f"Checkpoint MLP: {input_dim} -> {hidden_dim} -> {num_classes}")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Macro F1: {metrics['macro_f1']:.4f}")
    print(f"Weighted F1: {metrics['weighted_f1']:.4f}")
    print(f"Saved metrics: {metrics_path}")
    print(f"Saved predictions: {predictions_path}")


if __name__ == "__main__":
    main()
