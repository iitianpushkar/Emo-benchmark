#!/usr/bin/env python3
"""Train or evaluate a seven-class MOSEI MLP using saved VLM embeddings."""

from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path
if __package__ in (None, ""):
    _sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))

import argparse
import csv
import json
import random
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from scripts.data.prepare_mosei import EMOTIONS, SINGLE_LABELS, single_label


class MLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.ReLU(),
                                 nn.Dropout(dropout), nn.Linear(hidden_dim, len(SINGLE_LABELS)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def read_targets(path: Path, tie_policy: str) -> tuple[dict[str, int | None], dict[str, str], dict]:
    targets = {}
    label_status = {}
    statuses = Counter()
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        required = {"sample_id", *EMOTIONS}
        if not required <= set(reader.fieldnames or ()):
            raise ValueError(f"{path}: missing columns {sorted(required - set(reader.fieldnames or ()))}")
        for row in reader:
            sample_id = row["sample_id"]
            if sample_id in targets:
                raise ValueError(f"{path}: duplicate sample ID {sample_id}")
            label, status, _, _ = single_label(
                {name: float(row[name]) for name in EMOTIONS}, tie_policy, "neutral")
            statuses[status] += 1
            targets[sample_id] = None if label is None else SINGLE_LABELS.index(label)
            label_status[sample_id] = status
    return targets, label_status, {"indexed": len(targets), "label_status_counts": dict(statuses),
                                   "eligible": sum(value is not None for value in targets.values())}


def load_split(embedding_path: Path, index_path: Path, tie_policy: str):
    targets, label_status, index_info = read_targets(index_path, tie_policy)
    payload = torch.load(embedding_path, map_location="cpu", weights_only=False)
    x = payload["embeddings"].float()
    sample_ids = [str(value) for value in payload["sample_ids"]]
    if (x.ndim != 2 or x.shape[0] != len(sample_ids)
            or len(sample_ids) != len(set(sample_ids)) or not torch.isfinite(x).all()):
        raise ValueError(f"{embedding_path}: invalid embeddings or IDs")
    unknown = set(sample_ids) - targets.keys()
    if unknown:
        raise ValueError(f"{embedding_path}: IDs missing from index: {sorted(unknown)[:5]}")
    selected = [i for i, sample_id in enumerate(sample_ids) if targets[sample_id] is not None]
    if not selected:
        raise ValueError(f"{embedding_path}: no labeled embeddings")
    x = x[selected]
    ids = [sample_ids[i] for i in selected]
    y = torch.tensor([targets[sample_id] for sample_id in ids], dtype=torch.long)
    coverage = {**index_info, "embedded": len(ids),
                "missing_decoder": index_info["eligible"] - len(ids),
                "excluded_ties": index_info["indexed"] - index_info["eligible"],
                "embedded_label_status_counts": dict(Counter(label_status[sample_id] for sample_id in ids)),
                "embedded_class_counts": dict(Counter(SINGLE_LABELS[int(label)] for label in y)),
                "extractor_errors": len(payload.get("errors", []))}
    return x, y, ids, [label_status[sample_id] for sample_id in ids], coverage


def predict(model: MLP, x: torch.Tensor, device: torch.device, batch_size: int):
    model.eval()
    chunks = []
    with torch.inference_mode():
        for (batch,) in DataLoader(TensorDataset(x), batch_size=batch_size):
            chunks.append(torch.softmax(model(batch.to(device)), dim=-1).cpu())
    prob = torch.cat(chunks).numpy()
    return prob.argmax(axis=1), prob


def metrics(truth: np.ndarray, pred: np.ndarray) -> dict:
    labels = list(range(len(SINGLE_LABELS)))
    return {
        "sample_count": int(len(truth)),
        "accuracy": float(accuracy_score(truth, pred)),
        "macro_f1": float(f1_score(truth, pred, labels=labels, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(truth, pred, labels=labels, average="weighted", zero_division=0)),
        "classification_report": classification_report(
            truth, pred, labels=labels, target_names=SINGLE_LABELS,
            zero_division=0, output_dict=True),
        "confusion_matrix": confusion_matrix(truth, pred, labels=labels).tolist(),
        "class_order": list(SINGLE_LABELS),
    }


def save_predictions(path: Path, ids: list[str], statuses: list[str], truth: np.ndarray,
                     pred: np.ndarray, prob: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["sample_id", "label_status", "true_id", "true_label", "pred_id", "pred_label",
              "correct", "confidence", *(f"prob_{name}" for name in SINGLE_LABELS)]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for i, sample_id in enumerate(ids):
            row = {"sample_id": sample_id, "label_status": statuses[i],
                   "true_id": int(truth[i]),
                   "true_label": SINGLE_LABELS[int(truth[i])], "pred_id": int(pred[i]),
                   "pred_label": SINGLE_LABELS[int(pred[i])],
                   "correct": int(truth[i] == pred[i]), "confidence": float(prob[i, pred[i]])}
            row.update({f"prob_{name}": float(prob[i, j])
                        for j, name in enumerate(SINGLE_LABELS)})
            writer.writerow(row)


def save_confusion(path: Path, report: dict) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["true\\pred", *SINGLE_LABELS])
        for name, row in zip(SINGLE_LABELS, report["confusion_matrix"]):
            writer.writerow([name, *row])


def train(args: argparse.Namespace) -> None:
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available()
                          else "cpu" if args.device == "auto" else args.device)
    train_x, train_y, _, _, train_coverage = load_split(args.train_pt, args.train_index, args.tie_policy)
    valid_x, valid_y, valid_ids, valid_statuses, valid_coverage = load_split(
        args.valid_pt, args.valid_index, args.tie_policy)
    if train_x.shape[1] != valid_x.shape[1]:
        raise ValueError("Train and validation embedding dimensions differ")
    mean = train_x.mean(0)
    std = train_x.std(0).clamp_min(1e-6)
    train_x, valid_x = (train_x - mean) / std, (valid_x - mean) / std
    model = MLP(train_x.shape[1], args.hidden_dim, args.dropout).to(device)
    counts = torch.bincount(train_y, minlength=len(SINGLE_LABELS)).float()
    if (counts == 0).any():
        raise ValueError("At least one emotion class has no training examples")
    class_weights = counts.sum() / counts
    class_weights /= class_weights.mean()
    loss_fn = nn.CrossEntropyLoss(weight=class_weights.to(device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate,
                                  weight_decay=args.weight_decay)
    loader = DataLoader(TensorDataset(train_x, train_y), batch_size=args.batch_size, shuffle=True)
    best_f1, best_epoch, best_state = -1.0, 0, None
    for epoch in range(1, args.epochs + 1):
        model.train()
        for x, y in loader:
            optimizer.zero_grad(set_to_none=True)
            loss_fn(model(x.to(device)), y.to(device)).backward()
            optimizer.step()
        valid_pred, _ = predict(model, valid_x, device, args.batch_size)
        valid_f1 = f1_score(valid_y.numpy(), valid_pred,
                            labels=list(range(len(SINGLE_LABELS))),
                            average="macro", zero_division=0)
        print(f"epoch={epoch} valid_macro_f1={valid_f1:.4f}", flush=True)
        if valid_f1 > best_f1:
            best_f1, best_epoch = float(valid_f1), epoch
            best_state = {key: value.detach().cpu().clone()
                          for key, value in model.state_dict().items()}
    if best_state is None:
        raise RuntimeError("No checkpoint selected")
    model.load_state_dict(best_state)
    valid_pred, valid_prob = predict(model, valid_x, device, args.batch_size)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": best_state, "mean": mean, "std": std,
                "input_dim": train_x.shape[1], "hidden_dim": args.hidden_dim,
                "dropout": args.dropout, "best_epoch": best_epoch,
                "labels": list(SINGLE_LABELS), "tie_policy": args.tie_policy,
                "class_weights": class_weights.tolist()}, args.output_dir / "best_mlp.pt")
    report = {"task": "single_max_seven_class", "tie_policy": args.tie_policy,
              "zero_policy": "neutral", "best_epoch": best_epoch,
              "selection_metric": "validation_macro_f1", "coverage": {
                  "train": train_coverage, "validation": valid_coverage},
              "validation": metrics(valid_y.numpy(), valid_pred)}
    save_predictions(args.output_dir / "validation_predictions.csv", valid_ids, valid_statuses,
                     valid_y.numpy(), valid_pred, valid_prob)
    save_confusion(args.output_dir / "validation_confusion_matrix.csv", report["validation"])
    if args.test_pt:
        test_x, test_y, test_ids, test_statuses, test_coverage = load_split(
            args.test_pt, args.test_index, args.tie_policy)
        if test_x.shape[1] != train_x.shape[1]:
            raise ValueError("Test embedding dimension differs")
        test_pred, test_prob = predict(model, (test_x - mean) / std, device, args.batch_size)
        report["coverage"]["test"] = test_coverage
        report["test"] = metrics(test_y.numpy(), test_pred)
        save_predictions(args.output_dir / "test_predictions.csv", test_ids, test_statuses,
                         test_y.numpy(), test_pred, test_prob)
        save_confusion(args.output_dir / "test_confusion_matrix.csv", report["test"])
    (args.output_dir / "metrics.json").write_text(json.dumps(report, indent=2) + "\n")


def evaluate(args: argparse.Namespace) -> None:
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if tuple(checkpoint["labels"]) != SINGLE_LABELS:
        raise ValueError("Checkpoint class order differs")
    x, y, ids, statuses, coverage = load_split(
        args.embeddings_pt, args.index, checkpoint["tie_policy"])
    if x.shape[1] != checkpoint["input_dim"]:
        raise ValueError("Embedding dimension differs from checkpoint")
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available()
                          else "cpu" if args.device == "auto" else args.device)
    model = MLP(checkpoint["input_dim"], checkpoint["hidden_dim"], checkpoint["dropout"]).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    pred, prob = predict(model, (x - checkpoint["mean"]) / checkpoint["std"],
                         device, args.batch_size)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    save_predictions(args.output_dir / "predictions.csv", ids, statuses, y.numpy(), pred, prob)
    report = metrics(y.numpy(), pred)
    save_confusion(args.output_dir / "confusion_matrix.csv", report)
    (args.output_dir / "metrics.json").write_text(json.dumps({
        "task": "single_max_seven_class", "tie_policy": checkpoint["tie_policy"],
        "zero_policy": "neutral", "coverage": coverage,
        "metrics": report}, indent=2) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    training = sub.add_parser("train")
    for name in ("train", "valid"):
        training.add_argument(f"--{name}-pt", type=Path, required=True)
        training.add_argument(f"--{name}-index", type=Path, required=True)
    training.add_argument("--test-pt", type=Path)
    training.add_argument("--test-index", type=Path)
    training.add_argument("--tie-policy", choices=("first", "exclude"), default="exclude")
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
