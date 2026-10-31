#!/usr/bin/env python3
"""Train a six-label CMU-MOSEI transcript baseline with dev-selected thresholds."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score

EMOTIONS = ("happy", "sad", "anger", "surprise", "disgust", "fear")


def read_index(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"Empty index: {path}")
    return rows


def labels(rows: list[dict[str, str]]) -> np.ndarray:
    return np.asarray([[int(row[f"{name}_label"]) for name in EMOTIONS] for row in rows], dtype=np.int64)


def scores(y: np.ndarray, prob: np.ndarray, thresholds: np.ndarray) -> dict:
    pred = (prob >= thresholds).astype(np.int64)
    per_label = {}
    for j, name in enumerate(EMOTIONS):
        per_label[name] = {
            "support": int(y[:, j].sum()),
            "threshold": float(thresholds[j]),
            "precision": float(precision_score(y[:, j], pred[:, j], zero_division=0)),
            "recall": float(recall_score(y[:, j], pred[:, j], zero_division=0)),
            "f1": float(f1_score(y[:, j], pred[:, j], zero_division=0)),
            "roc_auc": float(roc_auc_score(y[:, j], prob[:, j])) if len(np.unique(y[:, j])) == 2 else None,
        }
    return {
        "sample_count": len(y),
        "macro_f1": float(f1_score(y, pred, average="macro", zero_division=0)),
        "micro_f1": float(f1_score(y, pred, average="micro", zero_division=0)),
        "subset_accuracy": float(accuracy_score(y, pred)),
        "per_label": per_label,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index-dir", type=Path, default=Path("data/indexes"))
    parser.add_argument("--output-dir", type=Path, default=Path("experiments/baselines/mosei_tfidf_text/outputs"))
    args = parser.parse_args()
    data = {split: read_index(args.index_dir / f"mosei_{split}_index.csv") for split in ("train", "valid", "test")}
    y = {split: labels(rows) for split, rows in data.items()}
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=100_000, sublinear_tf=True)
    x_train = vectorizer.fit_transform(row["utterance"] for row in data["train"])
    x_valid = vectorizer.transform(row["utterance"] for row in data["valid"])
    x_test = vectorizer.transform(row["utterance"] for row in data["test"])
    valid_prob = np.empty((len(data["valid"]), len(EMOTIONS)), dtype=np.float64)
    test_prob = np.empty((len(data["test"]), len(EMOTIONS)), dtype=np.float64)
    for j, name in enumerate(EMOTIONS):
        model = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)
        model.fit(x_train, y["train"][:, j])
        valid_prob[:, j] = model.predict_proba(x_valid)[:, 1]
        test_prob[:, j] = model.predict_proba(x_test)[:, 1]
        print(f"Fitted {name}", flush=True)
    grid = np.arange(0.05, 0.951, 0.025)
    thresholds = np.asarray([
        max(grid, key=lambda t: (f1_score(y["valid"][:, j], valid_prob[:, j] >= t, zero_division=0), -abs(t - 0.5)))
        for j in range(len(EMOTIONS))
    ])
    results = {
        "method": "TF-IDF word 1-2 grams, class-balanced logistic regression per emotion",
        "selection": "Per-emotion F1 thresholds selected on validation split only",
        "labels": list(EMOTIONS),
        "validation": scores(y["valid"], valid_prob, thresholds),
        "test": scores(y["test"], test_prob, thresholds),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "metrics.json").write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    with (args.output_dir / "test_predictions.csv").open("w", newline="", encoding="utf-8") as handle:
        fields = ["sample_id", *(f"true_{e}" for e in EMOTIONS), *(f"prob_{e}" for e in EMOTIONS), *(f"pred_{e}" for e in EMOTIONS)]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for i, row in enumerate(data["test"]):
            item = {"sample_id": row["sample_id"]}
            for j, name in enumerate(EMOTIONS):
                item[f"true_{name}"] = int(y["test"][i, j])
                item[f"prob_{name}"] = float(test_prob[i, j])
                item[f"pred_{name}"] = int(test_prob[i, j] >= thresholds[j])
            writer.writerow(item)
    print(json.dumps({split: {k: results[split][k] for k in ("sample_count", "macro_f1", "micro_f1", "subset_accuracy")} for split in ("validation", "test")}, indent=2))


if __name__ == "__main__":
    main()
