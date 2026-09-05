#!/usr/bin/env python3
"""Summarize alpha sweep MLP metrics."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


METRICS = ["accuracy", "macro_f1", "weighted_f1"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize outputs from run_alpha_sweep.py.")
    parser.add_argument("--output-root", type=Path, required=True, help="Root containing alpha_* MLP output directories.")
    parser.add_argument("--output-csv", type=Path, default=None, help="Optional CSV file to write.")
    return parser.parse_args()


def alpha_from_dir(path: Path) -> float:
    return float(path.name.removeprefix("alpha_").replace("_", "."))


def load_metrics(path: Path) -> dict[str, Any]:
    metrics_path = path / "metrics.json"
    if metrics_path.exists():
        with metrics_path.open(encoding="utf-8") as f:
            return json.load(f)

    dev_metrics_path = path / "dev_metrics.json"
    test_metrics_path = path / "test_metrics.json"
    if not dev_metrics_path.exists():
        raise FileNotFoundError(f"Could not find {metrics_path} or {dev_metrics_path}")

    metrics = {}
    with dev_metrics_path.open(encoding="utf-8") as f:
        metrics["best_dev"] = json.load(f)
    if test_metrics_path.exists():
        with test_metrics_path.open(encoding="utf-8") as f:
            metrics["test"] = json.load(f)
    return metrics


def split_metric(metrics: dict[str, Any], split: str, metric: str) -> float | None:
    if split not in metrics:
        return None
    return float(metrics[split][metric])


def format_float(value: float | None) -> str:
    return "" if value is None else f"{value:.4f}"


def main() -> None:
    args = parse_args()
    output_root = args.output_root.expanduser().resolve()
    alpha_dirs = sorted(output_root.glob("alpha_*"), key=alpha_from_dir)
    if not alpha_dirs:
        raise FileNotFoundError(f"No alpha_* directories found under {output_root}")

    rows = []
    for alpha_dir in alpha_dirs:
        metrics = load_metrics(alpha_dir)
        alpha = alpha_from_dir(alpha_dir)
        row = {
            "alpha_video": f"{alpha:.2f}",
            "beta_text": f"{1.0 - alpha:.2f}",
            "dev_accuracy": format_float(split_metric(metrics, "best_dev", "accuracy")),
            "dev_macro_f1": format_float(split_metric(metrics, "best_dev", "macro_f1")),
            "dev_weighted_f1": format_float(split_metric(metrics, "best_dev", "weighted_f1")),
            "test_accuracy": format_float(split_metric(metrics, "test", "accuracy")),
            "test_macro_f1": format_float(split_metric(metrics, "test", "macro_f1")),
            "test_weighted_f1": format_float(split_metric(metrics, "test", "weighted_f1")),
            "output_dir": str(alpha_dir),
        }
        rows.append(row)

    best_dev = max(rows, key=lambda row: float(row["dev_macro_f1"] or "-1"))
    best_test = max(rows, key=lambda row: float(row["test_macro_f1"] or "-1"))

    fieldnames = list(rows[0].keys())
    print(",".join(fieldnames))
    for row in rows:
        print(",".join(row[field] for field in fieldnames))
    print()
    print(f"Best dev macro F1: alpha={best_dev['alpha_video']} beta={best_dev['beta_text']} f1={best_dev['dev_macro_f1']}")
    if best_test["test_macro_f1"]:
        print(
            f"Best test macro F1: alpha={best_test['alpha_video']} "
            f"beta={best_test['beta_text']} f1={best_test['test_macro_f1']}"
        )

    if args.output_csv:
        args.output_csv.parent.mkdir(parents=True, exist_ok=True)
        with args.output_csv.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        print(f"Saved summary: {args.output_csv}")


if __name__ == "__main__":
    main()
