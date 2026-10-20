#!/usr/bin/env python3
"""Recompute MELD test metrics on one exact sample-ID intersection."""

from __future__ import annotations

# Allow both `python scripts/<group>/<file>.py` and `python -m scripts...`.
import sys as _sys
from pathlib import Path as _Path
if __package__ in (None, ""):
    _sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))


import argparse
import io
import zipfile
from dataclasses import dataclass
from pathlib import Path

import pandas as pd


@dataclass(frozen=True)
class ResultSpec:
    experiment: str
    method: str
    source: str
    member: str | None = None
    prediction_column: str = "pred_id"
    family: str = "main"


def classification_metrics(true: pd.Series, predicted: pd.Series) -> tuple[float, float, float]:
    labels = sorted(set(true) | set(predicted))
    per_class_f1 = []
    supports = []
    for label in labels:
        true_positive = int(((true == label) & (predicted == label)).sum())
        false_positive = int(((true != label) & (predicted == label)).sum())
        false_negative = int(((true == label) & (predicted != label)).sum())
        support = int((true == label).sum())
        denominator = 2 * true_positive + false_positive + false_negative
        per_class_f1.append(0.0 if denominator == 0 else 2 * true_positive / denominator)
        supports.append(support)

    accuracy = float((true == predicted).mean())
    macro_f1 = sum(per_class_f1) / len(per_class_f1)
    weighted_f1 = sum(score * support for score, support in zip(per_class_f1, supports)) / sum(supports)
    return accuracy, macro_f1, weighted_f1


def read_predictions(root: Path, spec: ResultSpec) -> pd.DataFrame:
    source = root / spec.source
    if spec.member is None:
        frame = pd.read_csv(source)
    else:
        with zipfile.ZipFile(source) as archive:
            with archive.open(spec.member) as handle:
                frame = pd.read_csv(io.BytesIO(handle.read()))

    required = {"sample_id", "true_id", spec.prediction_column}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"{spec.method}: missing columns {sorted(missing)}")
    if frame["sample_id"].duplicated().any():
        raise ValueError(f"{spec.method}: duplicate sample IDs")

    return frame[["sample_id", "true_id", spec.prediction_column]].rename(
        columns={spec.prediction_column: "pred_id"}
    )


def specs() -> list[ResultSpec]:
    baseline = "emo-benchmark-shared-results/outputs/qwen2_5_vl_3b_shared/test_predictions.csv"
    rows = [ResultSpec("baseline", "Full baseline", baseline)]

    pruning = [
        ("divprune_070", "DivPrune 0.70", "divprune_exp_0.7", "qwen2_5_vl_3b_divprune_070"),
        ("divprune_030", "DivPrune 0.30", "divprune_exp_0.3", "qwen2_5_vl_3b_divprune_070"),
        ("divprune_015", "DivPrune 0.15", "divprune_exp_0.15", "qwen2_5_vl_3b_divprune_015"),
        ("divprune_005", "DivPrune 0.05", "divprune_exp_0.05", "qwen2_5_vl_3b_divprune_005"),
        ("random_030", "Random tokens 0.30", "random_prune_0.3", "qwen2_5_vl_3b_random_prune_030_seed42"),
        ("random_005", "Random tokens 0.05", "random_prune_0.05", "qwen2_5_vl_3b_random_prune_005_seed40"),
    ]
    for experiment, method, archive, directory in pruning:
        for classifier, label in (("new_mlp", "new MLP"), ("previous_mlp", "previous MLP")):
            rows.append(
                ResultSpec(
                    experiment,
                    f"{method} + {label}",
                    archive,
                    f"outputs/{directory}/{classifier}/test_predictions.csv",
                    family="pruning",
                )
            )

    frame_dir = "outputs/qwen2_5_vl_3b_dinov2_base_frames_020"
    for classifier, label in (("new_mlp", "new MLP"), ("previous_mlp", "previous MLP")):
        rows.append(
            ResultSpec(
                "frame_020",
                f"DINOv2 diverse frames 0.20 + {label}",
                "frame_pruning.zip",
                f"{frame_dir}/{classifier}/test_predictions.csv",
                family="pruning",
            )
        )

    rows.extend(
        [
            ResultSpec(
                "visual_ablation",
                "Zero visual contribution",
                "visual_ablation.zip",
                "outputs/qwen2_5_vl_3b_visual_ablation/test_visual_ablation_predictions.csv",
                "zero_video_pred_id",
                "ablation",
            ),
            ResultSpec(
                "utterance_ablation",
                "Zero utterance contribution",
                "utterance_ablation.zip",
                "outputs/qwen2_5_vl_3b_utterance_ablation/test_utterance_ablation_predictions.csv",
                "zero_utterance_pred_id",
                "ablation",
            ),
        ]
    )

    for alpha in range(0, 101, 10):
        tag = f"alpha_{alpha / 100:.2f}".replace(".", "_")
        rows.append(
            ResultSpec(
                tag,
                f"Fusion video alpha {alpha / 100:.2f}",
                "alpha_sweep.zip",
                f"outputs/qwen2_5_vl_3b_fused_alpha_fixed_shared_mlp/{tag}/test_predictions.csv",
                family="alpha_sweep",
            )
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-root", type=Path, default=Path.home() / "Downloads/results")
    parser.add_argument("--output-dir", type=Path, default=Path("reports/compiled_experiments"))
    args = parser.parse_args()

    result_specs = specs()
    predictions = {spec.method: read_predictions(args.results_root, spec) for spec in result_specs}
    common_ids = set.intersection(*(set(frame["sample_id"]) for frame in predictions.values()))
    common_ids = sorted(common_ids)
    if not common_ids:
        raise ValueError("No sample IDs are shared by every result")

    baseline_ids = set(predictions["Full baseline"]["sample_id"])
    audit_rows = []
    metric_rows = []
    baseline_truth = None
    for spec in result_specs:
        frame = predictions[spec.method]
        ids = set(frame["sample_id"])
        aligned = frame[frame["sample_id"].isin(common_ids)].sort_values("sample_id")
        if len(aligned) != len(common_ids):
            raise AssertionError(f"{spec.method}: failed global alignment")
        if baseline_truth is None:
            baseline_truth = aligned[["sample_id", "true_id"]].reset_index(drop=True)
        else:
            truth = aligned[["sample_id", "true_id"]].reset_index(drop=True)
            if not truth.equals(baseline_truth):
                raise ValueError(f"{spec.method}: true labels disagree with baseline")

        audit_rows.append(
            {
                "experiment": spec.experiment,
                "method": spec.method,
                "family": spec.family,
                "native_n": len(frame),
                "unique_ids": len(ids),
                "overlap_with_baseline": len(ids & baseline_ids),
                "missing_from_baseline": len(baseline_ids - ids),
                "extra_vs_baseline": len(ids - baseline_ids),
                "global_intersection_n": len(common_ids),
            }
        )
        accuracy, macro_f1, weighted_f1 = classification_metrics(aligned["true_id"], aligned["pred_id"])
        metric_rows.append(
            {
                "experiment": spec.experiment,
                "method": spec.method,
                "family": spec.family,
                "n": len(aligned),
                "accuracy": accuracy,
                "macro_f1": macro_f1,
                "weighted_f1": weighted_f1,
            }
        )

    metrics = pd.DataFrame(metric_rows)
    baseline = metrics.loc[metrics["method"] == "Full baseline"].iloc[0]
    deltas = metrics.copy()
    for column in ("accuracy", "macro_f1", "weighted_f1"):
        deltas[f"delta_{column}"] = deltas[column] - baseline[column]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"sample_id": common_ids}).to_csv(args.output_dir / "aligned_test_sample_ids.csv", index=False)
    pd.DataFrame(audit_rows).to_csv(args.output_dir / "sample_alignment_audit.csv", index=False, float_format="%.6f")
    metrics.to_csv(args.output_dir / "aligned_test_metrics.csv", index=False, float_format="%.6f")
    deltas.to_csv(args.output_dir / "aligned_test_deltas.csv", index=False, float_format="%.6f")

    print(f"Globally aligned test samples: {len(common_ids)}")
    print(metrics[metrics["family"].isin(["main", "pruning"])].to_string(index=False, float_format=lambda x: f"{x:.6f}"))


if __name__ == "__main__":
    main()
