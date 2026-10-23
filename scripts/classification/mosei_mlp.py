#!/usr/bin/env python3
"""Train or evaluate a six-label CMU-MOSEI MLP on frozen VLM embeddings."""

from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

EMOTIONS = ("happy", "sad", "anger", "surprise", "disgust", "fear")


class MLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.net = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden_dim, 6))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def read_index(path: Path) -> dict[str, np.ndarray]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    result = {}
    for row in rows:
        sample_id = row["sample_id"]
        if sample_id in result:
            raise ValueError(f"Duplicate index sample_id: {sample_id}")
        result[sample_id] = np.asarray([int(row[f"{name}_label"]) for name in EMOTIONS], dtype=np.float32)
    return result


def load_split(embedding_path: Path, index_path: Path) -> tuple[torch.Tensor, torch.Tensor, list[str], dict]:
    payload = torch.load(embedding_path, map_location="cpu", weights_only=False)
    x = payload["embeddings"].float()
    ids = [str(value) for value in payload["sample_ids"]]
    targets = read_index(index_path)
    if x.ndim != 2 or len(ids) != len(x) or len(set(ids)) != len(ids):
        raise ValueError(f"Invalid embedding shape or sample IDs: {embedding_path}")
    if not torch.isfinite(x).all():
        raise ValueError(f"Non-finite embeddings: {embedding_path}")
    unknown = set(ids) - targets.keys()
    if unknown:
        raise ValueError(f"Embedding IDs absent from {index_path}: {sorted(unknown)[:5]}")
    y = torch.from_numpy(np.stack([targets[sample_id] for sample_id in ids]))
    return x, y, ids, {"indexed": len(targets), "embedded": len(ids), "missing": len(targets) - len(ids), "extractor_errors": len(payload.get("errors", []))}


def predict(model: MLP, x: torch.Tensor, device: torch.device, batch_size: int) -> np.ndarray:
    model.eval()
    values = []
    with torch.inference_mode():
        for (batch,) in DataLoader(TensorDataset(x), batch_size=batch_size):
            values.append(torch.sigmoid(model(batch.to(device))).cpu())
    return torch.cat(values).numpy()


def select_thresholds(y: np.ndarray, prob: np.ndarray) -> np.ndarray:
    grid = np.arange(0.05, 0.951, 0.025)
    return np.asarray([
        max(grid, key=lambda t: (f1_score(y[:, j], prob[:, j] >= t, zero_division=0), -abs(t - 0.5)))
        for j in range(6)
    ])


def metrics(y: np.ndarray, prob: np.ndarray, thresholds: np.ndarray) -> dict:
    pred = (prob >= thresholds).astype(np.int64)
    return {
        "sample_count": len(y),
        "macro_f1": float(f1_score(y, pred, average="macro", zero_division=0)),
        "micro_f1": float(f1_score(y, pred, average="micro", zero_division=0)),
        "subset_accuracy": float(accuracy_score(y, pred)),
        "per_label": {
            name: {
                "support": int(y[:, j].sum()),
                "threshold": float(thresholds[j]),
                "f1": float(f1_score(y[:, j], pred[:, j], zero_division=0)),
                "roc_auc": float(roc_auc_score(y[:, j], prob[:, j])) if len(np.unique(y[:, j])) == 2 else None,
            }
            for j, name in enumerate(EMOTIONS)
        },
    }


def save_predictions(path: Path, ids: list[str], y: np.ndarray, prob: np.ndarray, thresholds: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        fields = ["sample_id", *(f"true_{e}" for e in EMOTIONS), *(f"prob_{e}" for e in EMOTIONS), *(f"pred_{e}" for e in EMOTIONS)]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for i, sample_id in enumerate(ids):
            row = {"sample_id": sample_id}
            for j, name in enumerate(EMOTIONS):
                row[f"true_{name}"] = int(y[i, j])
                row[f"prob_{name}"] = float(prob[i, j])
                row[f"pred_{name}"] = int(prob[i, j] >= thresholds[j])
            writer.writerow(row)


def train(args: argparse.Namespace) -> None:
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else "cpu" if args.device == "auto" else args.device)
    train_x, train_y, _, train_coverage = load_split(args.train_pt, args.train_index)
    valid_x, valid_y, valid_ids, valid_coverage = load_split(args.valid_pt, args.valid_index)
    if train_x.shape[1] != valid_x.shape[1]:
        raise ValueError("Train/validation embedding dimensions differ")
    mean = train_x.mean(0)
    std = train_x.std(0).clamp_min(1e-6)
    train_x = (train_x - mean) / std
    valid_x = (valid_x - mean) / std
    model = MLP(train_x.shape[1], args.hidden_dim, args.dropout).to(device)
    positives = train_y.sum(0)
    pos_weight = ((len(train_y) - positives) / positives.clamp_min(1)).to(device)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    loader = DataLoader(TensorDataset(train_x, train_y), batch_size=args.batch_size, shuffle=True)
    best_f1 = -1.0
    best_state = None
    best_epoch = 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        for x, y in loader:
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(model(x.to(device)), y.to(device))
            loss.backward()
            optimizer.step()
        valid_prob = predict(model, valid_x, device, args.batch_size)
        valid_f1 = f1_score(valid_y.numpy(), valid_prob >= 0.5, average="macro", zero_division=0)
        print(f"epoch={epoch} valid_macro_f1_at_0.5={valid_f1:.4f}", flush=True)
        if valid_f1 > best_f1:
            best_f1 = valid_f1
            best_epoch = epoch
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    if best_state is None:
        raise RuntimeError("No checkpoint selected")
    model.load_state_dict(best_state)
    valid_prob = predict(model, valid_x, device, args.batch_size)
    thresholds = select_thresholds(valid_y.numpy(), valid_prob)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": best_state, "mean": mean, "std": std, "thresholds": thresholds.tolist(), "input_dim": train_x.shape[1], "hidden_dim": args.hidden_dim, "dropout": args.dropout, "best_epoch": best_epoch, "labels": list(EMOTIONS)}, args.output_dir / "best_mlp.pt")
    report = {"best_epoch": best_epoch, "checkpoint_macro_f1_at_0.5": float(best_f1), "coverage": {"train": train_coverage, "validation": valid_coverage}, "validation": metrics(valid_y.numpy(), valid_prob, thresholds)}
    save_predictions(args.output_dir / "validation_predictions.csv", valid_ids, valid_y.numpy(), valid_prob, thresholds)
    if args.test_pt:
        test_x, test_y, test_ids, test_coverage = load_split(args.test_pt, args.test_index)
        if test_x.shape[1] != train_x.shape[1]:
            raise ValueError("Test embedding dimension differs")
        test_prob = predict(model, (test_x - mean) / std, device, args.batch_size)
        report["coverage"]["test"] = test_coverage
        report["test"] = metrics(test_y.numpy(), test_prob, thresholds)
        save_predictions(args.output_dir / "test_predictions.csv", test_ids, test_y.numpy(), test_prob, thresholds)
    (args.output_dir / "metrics.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


def evaluate(args: argparse.Namespace) -> None:
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else "cpu" if args.device == "auto" else args.device)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if tuple(checkpoint["labels"]) != EMOTIONS:
        raise ValueError("Checkpoint emotion labels differ")
    x, y, ids, coverage = load_split(args.embeddings_pt, args.index)
    if x.shape[1] != checkpoint["input_dim"]:
        raise ValueError("Embedding dimension differs from checkpoint")
    model = MLP(checkpoint["input_dim"], checkpoint["hidden_dim"], checkpoint["dropout"]).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    prob = predict(model, (x - checkpoint["mean"]) / checkpoint["std"], device, args.batch_size)
    thresholds = np.asarray(checkpoint["thresholds"])
    args.output_dir.mkdir(parents=True, exist_ok=True)
    save_predictions(args.output_dir / "predictions.csv", ids, y.numpy(), prob, thresholds)
    (args.output_dir / "metrics.json").write_text(json.dumps({"coverage": coverage, "metrics": metrics(y.numpy(), prob, thresholds)}, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    training = sub.add_parser("train")
    for name in ("train", "valid"):
        training.add_argument(f"--{name}-pt", type=Path, required=True)
        training.add_argument(f"--{name}-index", type=Path, required=True)
    training.add_argument("--test-pt", type=Path)
    training.add_argument("--test-index", type=Path)
    training.add_argument("--hidden-dim", type=int, default=512)
    training.add_argument("--dropout", type=float, default=0.3)
    training.add_argument("--epochs", type=int, default=50)
    training.add_argument("--learning-rate", type=float, default=3e-4)
    training.add_argument("--weight-decay", type=float, default=1e-4)
    training.add_argument("--seed", type=int, default=42)
    evaluation = sub.add_parser("evaluate")
    evaluation.add_argument("--checkpoint", type=Path, required=True)
    evaluation.add_argument("--embeddings-pt", type=Path, required=True)
    evaluation.add_argument("--index", type=Path, required=True)
    for command in (training, evaluation):
        command.add_argument("--output-dir", type=Path, required=True)
        command.add_argument("--batch-size", type=int, default=64)
        command.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    args = parser.parse_args()
    if args.command == "train":
        if bool(args.test_pt) != bool(args.test_index):
            parser.error("--test-pt and --test-index must be supplied together")
        train(args)
    else:
        evaluate(args)


if __name__ == "__main__":
    main()
