#!/usr/bin/env python3
"""Train an MLP emotion classifier on frozen VLM embeddings."""

from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm.auto import tqdm

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
    parser = argparse.ArgumentParser(description="Train MLP on extracted emotion embeddings.")
    parser.add_argument("--train-pt", type=Path, required=True, help="Training embeddings .pt file.")
    parser.add_argument("--dev-pt", type=Path, required=True, help="Development/validation embeddings .pt file.")
    parser.add_argument("--test-pt", type=Path, default=None, help="Optional test embeddings .pt file.")
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory to save model and metrics.")
    parser.add_argument("--hidden-dim", type=int, default=512, help="MLP hidden layer dimension.")
    parser.add_argument("--dropout", type=float, default=0.3, help="Dropout probability.")
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs.")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size.")
    parser.add_argument("--learning-rate", type=float, default=3e-4, help="AdamW learning rate.")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="AdamW weight decay.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument(
        "--no-class-weights",
        action="store_true",
        help="Disable inverse-frequency class weights in cross entropy.",
    )
    parser.add_argument(
        "--device",
        default="auto",
        choices=["auto", "cpu", "cuda", "mps"],
        help="Training device.",
    )
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def choose_device(device_arg: str) -> torch.device:
    if device_arg != "auto":
        return torch.device(device_arg)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_embedding_file(path: Path) -> dict[str, Any]:
    # weights_only=False is needed because our payload contains metadata lists/dicts.
    payload = torch.load(path, map_location="cpu", weights_only=False)
    required = ["embeddings", "labels"]
    missing = [key for key in required if key not in payload]
    if missing:
        raise ValueError(f"{path} missing keys: {missing}")
    return payload


def get_xy(payload: dict[str, Any]) -> tuple[torch.Tensor, torch.Tensor]:
    x = payload["embeddings"].float()
    y = payload["labels"].long()
    if x.ndim != 2:
        raise ValueError(f"Expected embeddings shape [num_samples, dim], got {tuple(x.shape)}")
    if y.ndim != 1 or y.shape[0] != x.shape[0]:
        raise ValueError(f"Labels shape {tuple(y.shape)} does not match embeddings shape {tuple(x.shape)}")
    return x, y


def standardize_train_dev_test(
    x_train: torch.Tensor,
    x_dev: torch.Tensor,
    x_test: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Tensor, torch.Tensor]:
    mean = x_train.mean(dim=0, keepdim=True)
    std = x_train.std(dim=0, keepdim=True).clamp_min(1e-6)
    x_train = (x_train - mean) / std
    x_dev = (x_dev - mean) / std
    if x_test is not None:
        x_test = (x_test - mean) / std
    return x_train, x_dev, x_test, mean.squeeze(0), std.squeeze(0)


def make_class_weights(labels: torch.Tensor, num_classes: int) -> torch.Tensor:
    counts = torch.bincount(labels, minlength=num_classes).float()
    weights = counts.sum() / counts.clamp_min(1.0)
    weights = weights / weights.mean().clamp_min(1e-6)
    return weights


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
) -> float:
    is_train = optimizer is not None
    model.train(is_train)
    total_loss = 0.0
    total_count = 0

    for x, y in loader:
        x = x.to(device)
        y = y.to(device)

        if is_train:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(is_train):
            logits = model(x)
            loss = criterion(logits, y)

        if is_train:
            loss.backward()
            optimizer.step()

        total_loss += float(loss.item()) * x.shape[0]
        total_count += x.shape[0]

    return total_loss / max(total_count, 1)


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
    path.parent.mkdir(parents=True, exist_ok=True)

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
    set_seed(args.seed)
    device = choose_device(args.device)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    train_payload = load_embedding_file(args.train_pt)
    dev_payload = load_embedding_file(args.dev_pt)
    test_payload = load_embedding_file(args.test_pt) if args.test_pt else None

    label_names = train_payload.get("label_names", DEFAULT_LABELS)
    num_classes = len(label_names)

    x_train, y_train = get_xy(train_payload)
    x_dev, y_dev = get_xy(dev_payload)
    x_test, y_test = get_xy(test_payload) if test_payload else (None, None)

    x_train, x_dev, x_test, feature_mean, feature_std = standardize_train_dev_test(x_train, x_dev, x_test)

    input_dim = x_train.shape[1]
    model = EmotionMLP(input_dim=input_dim, hidden_dim=args.hidden_dim, num_classes=num_classes, dropout=args.dropout).to(device)

    class_weights = None if args.no_class_weights else make_class_weights(y_train, num_classes).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

    train_loader = DataLoader(TensorDataset(x_train, y_train), batch_size=args.batch_size, shuffle=True)
    dev_loader = DataLoader(TensorDataset(x_dev, y_dev), batch_size=args.batch_size, shuffle=False)

    print(f"Device: {device}")
    print(f"Train embeddings: {tuple(x_train.shape)}")
    print(f"Dev embeddings: {tuple(x_dev.shape)}")
    if x_test is not None:
        print(f"Test embeddings: {tuple(x_test.shape)}")
    print(f"MLP: {input_dim} -> {args.hidden_dim} -> {num_classes}")
    print(f"Class weights: {'disabled' if args.no_class_weights else class_weights.detach().cpu().tolist()}")

    history = []
    best_dev_macro_f1 = -1.0
    best_state = None

    for epoch in tqdm(range(1, args.epochs + 1), desc="epochs"):
        train_loss = run_epoch(model, train_loader, criterion, optimizer, device)
        dev_loss = run_epoch(model, dev_loader, criterion, None, device)

        dev_pred, _ = predict(model, x_dev, args.batch_size, device)
        dev_true = y_dev.numpy()
        dev_metrics = compute_metrics(dev_true, dev_pred, label_names)

        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "dev_loss": dev_loss,
            "dev_accuracy": dev_metrics["accuracy"],
            "dev_macro_f1": dev_metrics["macro_f1"],
            "dev_weighted_f1": dev_metrics["weighted_f1"],
        }
        history.append(row)

        if dev_metrics["macro_f1"] > best_dev_macro_f1:
            best_dev_macro_f1 = dev_metrics["macro_f1"]
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}

        print(
            f"epoch={epoch:03d} "
            f"train_loss={train_loss:.4f} dev_loss={dev_loss:.4f} "
            f"dev_acc={dev_metrics['accuracy']:.4f} "
            f"dev_macro_f1={dev_metrics['macro_f1']:.4f}"
        )

    if best_state is None:
        raise RuntimeError("No best model state was captured.")

    model.load_state_dict(best_state)

    dev_pred, dev_probs = predict(model, x_dev, args.batch_size, device)
    dev_metrics = compute_metrics(y_dev.numpy(), dev_pred, label_names)

    final_metrics = {"best_dev": dev_metrics}

    if x_test is not None and y_test is not None and test_payload is not None:
        test_pred, test_probs = predict(model, x_test, args.batch_size, device)
        test_metrics = compute_metrics(y_test.numpy(), test_pred, label_names)
        final_metrics["test"] = test_metrics
        write_predictions(
            args.output_dir / "test_predictions.csv",
            test_payload,
            y_test.numpy(),
            test_pred,
            test_probs,
            label_names,
        )

    write_predictions(
        args.output_dir / "dev_predictions.csv",
        dev_payload,
        y_dev.numpy(),
        dev_pred,
        dev_probs,
        label_names,
    )

    with (args.output_dir / "training_history.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(history[0].keys()))
        writer.writeheader()
        writer.writerows(history)

    checkpoint = {
        "model_state_dict": best_state,
        "input_dim": input_dim,
        "hidden_dim": args.hidden_dim,
        "dropout": args.dropout,
        "num_classes": num_classes,
        "label_names": label_names,
        "feature_mean": feature_mean,
        "feature_std": feature_std,
        "args": vars(args),
        "best_dev_macro_f1": best_dev_macro_f1,
    }
    torch.save(checkpoint, args.output_dir / "best_mlp.pt")

    with (args.output_dir / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump(final_metrics, f, indent=2)

    print(f"Saved best checkpoint: {args.output_dir / 'best_mlp.pt'}")
    print(f"Saved metrics: {args.output_dir / 'metrics.json'}")
    print("Best dev metrics:")
    print(json.dumps({k: dev_metrics[k] for k in ["accuracy", "macro_f1", "weighted_f1"]}, indent=2))


if __name__ == "__main__":
    main()
