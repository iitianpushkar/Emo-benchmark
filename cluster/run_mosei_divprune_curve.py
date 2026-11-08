#!/usr/bin/env python3
"""Run Qwen2.5-VL-7B visual-token retention curves on MOSEI."""

from __future__ import annotations

import argparse
import csv
import json
import os
import statistics
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from cluster.live_log import activate_live_log  # noqa: E402
from cluster.mosei_cache_subset import (cached_multilabel_rows, cached_single_label_rows,
                                        resolve_cache_frames, write_subset)  # noqa: E402
from cluster.preflight import verify_model, verify_mosei  # noqa: E402
from cluster.qwen_cost import FLOP_PROFILE_TARGET, summarize_embedding_cost  # noqa: E402
from cluster.resume_config import record_resume_config  # noqa: E402
from cluster.verify_qwen_environment import require_cuda_gpu, verify_qwen_environment  # noqa: E402
from cluster.run_mosei_qwen2_5vl7b_baseline import (  # noqa: E402
    HIDDEN_SIZE, MODEL_ID, PROMPT_VERSIONS, SPLIT_COUNTS, git_revision, index_ids, single_label_counts,
    run, sha256, verify_available_embeddings,
)

DEFAULT_RATIOS = (1.0, .8, .7, .6, .5, .4, .3, .2, .15, .1, .05)
SEVEN_CLASS_RATIOS = (.05, .1, .15, .2, .3, .5, .7)
DEFAULT_SEEDS = (42, 43, 44)
CURVE_PROMPT_VERSIONS = {**PROMPT_VERSIONS,
                         "meld_emotion_task": "meld_seven_class_emotion_task_v1"}


def ratio_name(ratio: float) -> str:
    percent = round(ratio * 100)
    if abs(ratio * 100 - percent) > 1e-8 or not 5 <= percent <= 100:
        raise ValueError("Ratios must be whole percentages from 5% to 100%")
    return f"r{percent:03d}"


def curve_ratios(requested: list[float] | None,
                 single_label_mode: bool,
                 emotion_multilabel_mode: bool = False) -> tuple[list[float], list[str]]:
    baseline_reference_mode = single_label_mode or emotion_multilabel_mode
    ratios = list(requested if requested is not None else
                  SEVEN_CLASS_RATIOS if baseline_reference_mode else DEFAULT_RATIOS)
    names = [ratio_name(ratio) for ratio in ratios]
    if len(names) != len(set(names)):
        raise ValueError("Ratios must be unique")
    if baseline_reference_mode and "r100" in names:
        raise ValueError("Emotion-prompt curves reserve 100% for the full baseline; omit 1.0")
    if not baseline_reference_mode and "r100" not in names:
        raise ValueError("Utterance-only curves must include 1.0 for their full baseline")
    return ratios, names


def complete_chunk(path: Path, ids: list[str], model_dir: Path, start: int,
                   ratio: float, protocol: dict, profile_stride: int,
                   method: str = "divprune", random_seed: int | None = None,
                   important_ratio: float = 0.5, fastv_k: int = 2,
                   prompt_style: str = "utterance_only") -> bool:
    if not path.is_file():
        return False
    import torch
    if method == "random":
        from scripts.pruning.random_prune import sample_seed

    payload = torch.load(path, map_location="cpu", weights_only=False)
    saved_ids = [str(value) for value in payload.get("sample_ids", [])]
    config = payload.get("config", {})
    expected_pruning = ((f"fastv_after_layer_{fastv_k}_video_tokens" if method == "fastv"
                         else "sparsevlm_progressive_decoder_video_tokens" if method == "sparsevlm"
                         else f"{method}_projected_video_tokens") if ratio < 1 else "none")
    return (
        not payload.get("errors") and len(saved_ids) == len(ids)
        and len(set(saved_ids)) == len(ids) and set(saved_ids) == set(ids)
        and tuple(payload["embeddings"].shape) == (len(ids), HIDDEN_SIZE)
        and torch.isfinite(payload["embeddings"]).all().item()
        and config.get("model_dir") == str(model_dir)
        and config.get("start") == start and config.get("limit") == len(ids)
        and config.get("retain_ratio") == ratio
        and config.get("pruning") == expected_pruning
        and config.get("random_seed") == random_seed
        and (method != "vispruner" or config.get("important_ratio") == important_ratio)
        and (method != "visionzip" or (
            config.get("attention_source") == "last_full_qwen_vision_block_received_attention_and_keys"
            and config.get("attention_score_reduction") == "mean_heads_sum_queries_per_attention_segment_v3"
            and config.get("merging") == "official_qwen_65_to_5_scaled_budget_with_temporal_group_isolation_v1"
        ))
        and (method != "fastv" or config.get("fastv_k") == fastv_k)
        and (method != "sparsevlm" or (
            config.get("sparsevlm_stages") == [2, 6, 15]
            and config.get("sparsevlm_schedule") == "matrix_rank_lambda_geometric_with_final_budget_cap_v1"
            and config.get("recycling") == "top_30_percent_discarded_then_density_cluster_mean"
        ))
        and config.get("profile_stride") == profile_stride
        and config.get("video_protocol") == protocol
        and config.get("prompt_style") == (
            CURVE_PROMPT_VERSIONS[prompt_style] if prompt_style != "utterance_only" else None)
        and len(payload.get("metadata", [])) == len(ids)
        and (method != "random" or all(
            item.get("random_prune_sample_seed") == sample_seed(random_seed, sample_id)
            for sample_id, item in zip(saved_ids, payload["metadata"])
        ))
    )


def decode_failure_records(payload: dict, expected_ids: list[str], model_dir: Path,
                           start: int, ratio: float, protocol: dict,
                           profile_stride: int, method: str,
                           random_seed: int | None, important_ratio: float,
                           fastv_k: int, prompt_style: str = "utterance_only") -> list[dict]:
    """Accept only fully attempted chunks whose failures are video decoding errors."""
    errors = payload.get("errors", [])
    saved_ids = [str(value) for value in payload.get("sample_ids", [])]
    error_ids = [error.get("sample_id") for error in errors]
    observed_ids = saved_ids + error_ids
    config = payload.get("config", {})
    expected_pruning = ((f"fastv_after_layer_{fastv_k}_video_tokens" if method == "fastv"
                         else "sparsevlm_progressive_decoder_video_tokens" if method == "sparsevlm"
                         else f"{method}_projected_video_tokens") if ratio < 1 else "none")
    if (not errors or len(observed_ids) != len(expected_ids)
            or len(set(observed_ids)) != len(expected_ids)
            or set(observed_ids) != set(expected_ids)
            or tuple(payload["embeddings"].shape) != (len(saved_ids), HIDDEN_SIZE)
            or len(payload.get("metadata", [])) != len(saved_ids)
            or config.get("model_dir") != str(model_dir)
            or config.get("start") != start or config.get("limit") != len(expected_ids)
            or config.get("video_protocol") != protocol
            or config.get("prompt_style") != (
                CURVE_PROMPT_VERSIONS[prompt_style] if prompt_style != "utterance_only" else None)
            or config.get("retain_ratio") != ratio
            or config.get("pruning") != expected_pruning
            or config.get("profile_stride") != profile_stride
            or config.get("random_seed") != random_seed
            or (method == "vispruner" and config.get("important_ratio") != important_ratio)
            or (method == "visionzip" and (
                config.get("attention_source") != "last_full_qwen_vision_block_received_attention_and_keys"
                or config.get("attention_score_reduction") != "mean_heads_sum_queries_per_attention_segment_v3"
                or config.get("merging") != "official_qwen_65_to_5_scaled_budget_with_temporal_group_isolation_v1"
            ))
            or (method == "fastv" and config.get("fastv_k") != fastv_k)
            or (method == "sparsevlm" and (
                config.get("sparsevlm_stages") != [2, 6, 15]
                or config.get("sparsevlm_schedule") != "matrix_rank_lambda_geometric_with_final_budget_cap_v1"
                or config.get("recycling") != "top_30_percent_discarded_then_density_cluster_mean"
            ))
            or any(not str(error.get("error", "")).startswith(
                ("TimeoutError: Decord exceeded", "RuntimeError: Video decoding failed"))
                for error in errors)):
        raise ValueError("Chunk failed for a reason other than completed decoder timeouts/errors")
    return errors


def training_metrics_match(path: Path, coverage: dict[str, dict[str, int]],
                           single_label_mode: bool = False) -> bool:
    if (not path.is_file() or not (path.parent / "best_mlp.pt").is_file()
            or not (path.parent / "validation_predictions.csv").is_file()):
        return False
    try:
        report = json.loads(path.read_text())
        if report.get("task") != ("single_max_seven_class" if single_label_mode else None):
            return False
        for split, report_split in (("train", "train"), ("valid", "validation")):
            observed = report["coverage"][report_split]
            expected = coverage[split]
            missing_key = "missing_decoder" if single_label_mode else "missing"
            if (observed["indexed"] != expected["indexed"]
                    or observed["embedded"] != expected["embedded"]
                    or observed[missing_key] != expected["missing"]):
                return False
        return report["validation"]["sample_count"] == coverage["valid"]["embedded"]
    except (OSError, ValueError, KeyError, TypeError):
        return False


def test_metrics_match(path: Path, coverage: dict[str, int],
                       single_label_mode: bool = False) -> bool:
    if not path.is_file() or not (path.parent / "predictions.csv").is_file():
        return False
    try:
        report = json.loads(path.read_text())
        if report.get("task") != ("single_max_seven_class" if single_label_mode else None):
            return False
        observed = report["coverage"]
        return (observed["indexed"] == coverage["indexed"]
                and observed["embedded"] == coverage["embedded"]
                and observed["missing_decoder" if single_label_mode else "missing"] == coverage["missing"]
                and report["metrics"]["sample_count"] == coverage["embedded"])
    except (OSError, ValueError, KeyError, TypeError):
        return False


def write_curve(root: Path, names: list[str], seeds: tuple[int, ...],
                method: str = "divprune", random_seed: int | None = None,
                single_label_mode: bool = False) -> None:
    rows = []
    important_ratio = (json.loads((root / "run_config.json").read_text())["important_ratio"]
                       if method == "vispruner" else None)
    fastv_k = (json.loads((root / "run_config.json").read_text())["fastv_k"]
               if method == "fastv" else None)
    sparsevlm_stages = (json.loads((root / "run_config.json").read_text())["sparsevlm_stages"]
                        if method == "sparsevlm" else None)
    for name in names:
        folder = root / name
        split_summaries = json.loads((folder / "cost_summary.json").read_text())
        coverage = json.loads((folder / "coverage.json").read_text())
        reports = [json.loads((folder / "mlp" / f"seed_{seed}" / "metrics.json").read_text())
                   for seed in seeds]
        if single_label_mode:
            if any(report.get("task") != "single_max_seven_class" for report in reports):
                raise ValueError(f"Expected seven-class MLP metrics in {folder}")
            validation_f1 = [float(report["validation"]["macro_f1"]) for report in reports]
            validation_accuracy = [float(report["validation"]["accuracy"]) for report in reports]
            validation_weighted_f1 = [float(report["validation"]["weighted_f1"]) for report in reports]
        else:
            checkpoint_f1 = [float(report["checkpoint_macro_f1_at_0.5"]) for report in reports]
            tuned_f1 = [float(report["validation"]["macro_f1"]) for report in reports]
            validation_accuracy = [float(report["validation"]["subset_accuracy"]) for report in reports]
            validation_micro_f1 = [float(report["validation"]["micro_f1"]) for report in reports]
            validation_weighted_accuracy = [report["validation"].get("mean_weighted_accuracy")
                                            for report in reports]
        row = {
            "method": method,
            "extraction_seed": random_seed,
            "important_ratio": important_ratio,
            "fastv_k": fastv_k,
            "sparsevlm_stages": sparsevlm_stages,
            "condition": name,
            "requested_retention": int(name[1:]) / 100,
            "train_indexed": coverage["train"]["indexed"],
            "train_embedded": coverage["train"]["embedded"],
            "train_missing": coverage["train"]["missing"],
            "valid_indexed": coverage["valid"]["indexed"],
            "valid_embedded": coverage["valid"]["embedded"],
            "valid_missing": coverage["valid"]["missing"],
            "validation_coverage_complete": all(coverage[split]["missing"] == 0
                                                for split in ("train", "valid")),
            "actual_retention_train": split_summaries["train"]["weighted_retention"],
            "actual_retention_valid": split_summaries["valid"]["weighted_retention"],
            "mean_full_visual_tokens_valid": split_summaries["valid"]["mean_full_visual_tokens_per_clip"],
            "mean_retained_visual_tokens_valid": split_summaries["valid"]["mean_retained_visual_tokens_per_clip"],
            "mean_retained_sequence_tokens_valid": split_summaries["valid"]["mean_retained_sequence_tokens_per_clip"],
            "valid_mean_end_to_end_seconds": split_summaries["valid"]["latency_seconds_per_clip"]["end_to_end_seconds"]["mean"],
            "valid_p95_end_to_end_seconds": split_summaries["valid"]["latency_seconds_per_clip"]["end_to_end_seconds"]["p95"],
            "valid_mean_forward_seconds": split_summaries["valid"]["latency_seconds_per_clip"]["forward_seconds"]["mean"],
            "valid_peak_total_allocated_gib_mean": split_summaries["valid"]["peak_total_allocated_gib"]["mean"],
            "valid_peak_total_reserved_gib_mean": split_summaries["valid"]["peak_total_reserved_gib"]["mean"],
            "valid_incremental_peak_allocated_gib_mean": split_summaries["valid"]["incremental_peak_allocated_gib"]["mean"],
            "valid_profiled_clips": split_summaries["valid"]["profiled_clips"],
            "valid_theoretical_forward_tflops_mean": split_summaries["valid"]["theoretical_forward_tflops_per_profiled_clip"]["mean"],
            "valid_theoretical_forward_tflops_sd": split_summaries["valid"]["theoretical_forward_tflops_per_profiled_clip"]["sd"],
        }
        if single_label_mode:
            row.update(valid_macro_f1_mean=statistics.mean(validation_f1),
                       valid_macro_f1_sd=statistics.stdev(validation_f1) if len(validation_f1) > 1 else 0.0,
                       valid_accuracy_mean=statistics.mean(validation_accuracy),
                       valid_weighted_f1_mean=statistics.mean(validation_weighted_f1))
        else:
            row.update({"valid_macro_f1_at_0.5_mean": statistics.mean(checkpoint_f1),
                        "valid_macro_f1_at_0.5_sd": statistics.stdev(checkpoint_f1) if len(checkpoint_f1) > 1 else 0.0,
                        "valid_tuned_macro_f1_mean": statistics.mean(tuned_f1),
                        "valid_tuned_macro_f1_sd": statistics.stdev(tuned_f1) if len(tuned_f1) > 1 else 0.0,
                        "valid_subset_accuracy_mean": statistics.mean(validation_accuracy),
                        "valid_micro_f1_mean": statistics.mean(validation_micro_f1)})
            defined_weighted = [float(value) for value in validation_weighted_accuracy
                                if value is not None]
            row["valid_mean_weighted_accuracy_mean"] = (statistics.mean(defined_weighted)
                                                          if defined_weighted else None)
        test_reports = [folder / "test_eval" / f"seed_{seed}" / "metrics.json" for seed in seeds]
        if "test" in coverage and all(path.is_file() for path in test_reports):
            row["test_indexed"] = coverage["test"]["indexed"]
            row["test_embedded"] = coverage["test"]["embedded"]
            row["test_missing"] = coverage["test"]["missing"]
            test_payloads = [json.loads(path.read_text()) for path in test_reports]
            test_metrics = [payload["metrics"] for payload in test_payloads]
            values = [float(metric["macro_f1"]) for metric in test_metrics]
            row["test_macro_f1_mean"] = statistics.mean(values)
            row["test_macro_f1_sd"] = statistics.stdev(values) if len(values) > 1 else 0.0
            if single_label_mode:
                row["test_accuracy_mean"] = statistics.mean(float(metric["accuracy"]) for metric in test_metrics)
                row["test_weighted_f1_mean"] = statistics.mean(float(metric["weighted_f1"]) for metric in test_metrics)
            else:
                row["test_micro_f1_mean"] = statistics.mean(float(metric["micro_f1"]) for metric in test_metrics)
                row["test_subset_accuracy_mean"] = statistics.mean(
                    float(metric["subset_accuracy"]) for metric in test_metrics)
                weighted = [float(metric["mean_weighted_accuracy"]) for metric in test_metrics
                            if metric.get("mean_weighted_accuracy") is not None]
                row["test_mean_weighted_accuracy_mean"] = (statistics.mean(weighted)
                                                             if weighted else None)
                if all("metrics_fixed_0_5" in payload for payload in test_payloads):
                    fixed_metrics = [payload["metrics_fixed_0_5"] for payload in test_payloads]
                    row["test_macro_f1_at_0.5_mean"] = statistics.mean(
                        float(metric["macro_f1"]) for metric in fixed_metrics)
        rows.append(row)
    rows.sort(key=lambda row: row["requested_retention"])
    f1_key = "valid_macro_f1_mean" if single_label_mode else "valid_macro_f1_at_0.5_mean"
    sd_key = "valid_macro_f1_sd" if single_label_mode else "valid_macro_f1_at_0.5_sd"
    baseline = next((row[f1_key] for row in rows if row["requested_retention"] == 1), None)
    if baseline is not None:
        for row in rows:
            row["delta_valid_macro_f1_at_0.5"] = row[f1_key] - baseline
    fieldnames = list(dict.fromkeys(key for row in rows for key in row))
    with (root / "validation_curve.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    (root / "validation_curve.json").write_text(json.dumps(rows, indent=2) + "\n")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(2, 2, figsize=(11, 8))
    x = [row["actual_retention_valid"] * 100 for row in rows]
    y = [row[f1_key] for row in rows]
    error = [row[sd_key] for row in rows]
    axes[0, 0].errorbar(x, y, yerr=error, marker="o", capsize=3)
    if baseline is not None:
        axes[0, 0].axhline(baseline, color="gray", linestyle="--", linewidth=1)
    axes[0, 0].set(xlabel="Measured visual-token retention (%)",
                ylabel="Seven-class validation macro F1" if single_label_mode else "Validation macro F1 at 0.5",
                title="Matched MLP, mean ± SD across seeds")
    axes[0, 1].plot(x, [row["valid_mean_end_to_end_seconds"] for row in rows], marker="o")
    axes[0, 1].set(xlabel="Measured visual-token retention (%)", ylabel="Seconds per validation clip",
                   title="End-to-end extraction latency")
    axes[1, 0].plot(x, [row["valid_peak_total_allocated_gib_mean"] for row in rows], marker="o")
    axes[1, 0].set(xlabel="Measured visual-token retention (%)", ylabel="Peak allocated GiB",
                   title="Process GPU allocation, model included")
    axes[1, 1].errorbar(
        x, [row["valid_theoretical_forward_tflops_mean"] for row in rows],
        yerr=[row["valid_theoretical_forward_tflops_sd"] for row in rows],
        marker="o", capsize=3,
    )
    axes[1, 1].set(xlabel="Measured visual-token retention (%)", ylabel="Theoretical TFLOPs/clip",
                   title="Shape-based forward FLOPs on profiled clips")
    for axis in axes.flat:
        axis.grid(alpha=0.25)
    figure.tight_layout()
    figure.savefig(root / "validation_curve.png", dpi=180)
    plt.close(figure)
    if all("test_macro_f1_mean" in row for row in rows):
        figure, axes = plt.subplots(2, 2, figsize=(11, 8))
        test_x = [row["requested_retention"] * 100 for row in rows]
        axes[0, 0].errorbar(test_x, [row["test_macro_f1_mean"] for row in rows],
                            yerr=[row["test_macro_f1_sd"] for row in rows],
                            marker="o", capsize=3)
        axes[0, 0].set(ylabel="Test macro F1", title="Mean ± SD across MLP seeds")
        if single_label_mode:
            panels = ((axes[0, 1], "test_accuracy_mean", "Test accuracy"),
                      (axes[1, 0], "test_weighted_f1_mean", "Test weighted F1"),
                      (axes[1, 1], None, "Test minus validation macro F1"))
        else:
            panels = ((axes[0, 1], "test_micro_f1_mean", "Test micro F1"),
                      (axes[1, 0], "test_subset_accuracy_mean", "Test subset accuracy"),
                      (axes[1, 1], "test_mean_weighted_accuracy_mean", "Test mean weighted accuracy"))
        for axis, key, label in panels:
            values = ([row[key] for row in rows] if key else
                      [row["test_macro_f1_mean"] - row["valid_macro_f1_mean"] for row in rows])
            axis.plot(test_x, values, marker="o")
            axis.set(ylabel=label)
        for axis in axes.flat:
            axis.set(xlabel="Requested visual-token retention (%)")
            axis.grid(alpha=0.25)
        figure.tight_layout()
        figure.savefig(root / "test_metrics_curve.png", dpi=180)
        plt.close(figure)
    print(f"Wrote {root / 'validation_curve.csv'}, validation_curve.png, "
          f"and test_metrics_curve.png", flush=True)


def main(method: str = "divprune") -> int:
    if method not in ("divprune", "random", "uniform", "vispruner", "visionzip", "fastv", "sparsevlm"):
        raise ValueError(f"Unknown token pruning method: {method}")
    parser = argparse.ArgumentParser(description=f"Run a Qwen2.5-VL-7B {method} token-retention curve on MOSEI")
    parser.add_argument("--root", type=Path, default=REPO_ROOT.parents[1])
    parser.add_argument("--model-dir", type=Path)
    parser.add_argument("--mosei-root", type=Path)
    parser.add_argument("--decoded-cache-dir", type=Path,
                        help="Read predecoded frames only; cache misses become recorded decode failures")
    parser.add_argument("--single-label-csv", type=Path,
                        help="Full seven-class MOSEI ground truth; with a cache, select valid cached clips")
    parser.add_argument("--prompt-style", choices=tuple(PROMPT_VERSIONS),
                        default="utterance_only")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--ratios", nargs="+", type=float,
                        help="Retention ratios; emotion-prompt modes default to 5,10,15,20,30,50,70%%")
    parser.add_argument("--seeds", nargs="+", type=int, default=list(DEFAULT_SEEDS))
    if method == "random":
        parser.add_argument("--random-seed", type=int, default=42,
                            help="Base seed for deterministic per-sample token selection")
    if method == "vispruner":
        parser.add_argument("--important-ratio", type=float, default=0.5,
                            help="Fraction of retained tokens chosen by vision attention")
    if method == "fastv":
        parser.add_argument("--fastv-k", type=int, default=2,
                            help="Number of full decoder layers before FastV pruning")
    parser.add_argument("--chunk-size", type=int, default=500)
    parser.add_argument("--log-every", type=int, default=1,
                        help="Print extraction progress every N new clips")
    parser.add_argument("--save-every", type=int, default=25,
                        help="Checkpoint embeddings every N new clips")
    parser.add_argument("--decode-timeout-seconds", type=float, default=10,
                        help="Skip clips whose Decord decode exceeds this many seconds")
    parser.add_argument("--verify-only", action="store_true")
    parser.add_argument("--test-all", action="store_true",
                        help="Compatibility flag; every run now evaluates test")
    args = parser.parse_args()
    random_seed = args.random_seed if method == "random" else None
    important_ratio = args.important_ratio if method == "vispruner" else 0.5
    fastv_k = args.fastv_k if method == "fastv" else 2
    if (args.chunk_size <= 0 or args.log_every <= 0 or args.save_every <= 0
            or args.decode_timeout_seconds <= 0 or not args.seeds
            or len(set(args.seeds)) != len(args.seeds)):
        parser.error("Chunk/log/save intervals must be positive and at least one unique seed is required")
    if random_seed is not None and random_seed < 0:
        parser.error("--random-seed must be nonnegative")
    single_label_mode = args.single_label_csv is not None
    if single_label_mode and args.prompt_style != "emotion_task":
        parser.error("--single-label-csv requires --prompt-style emotion_task")
    if not single_label_mode and args.prompt_style == "emotion_task":
        parser.error("--prompt-style emotion_task requires --single-label-csv; use emotion_multilabel_task for six emotions")
    if not 0 <= important_ratio <= 1:
        parser.error("--important-ratio must be in [0, 1]")
    if fastv_k < 1:
        parser.error("--fastv-k must be positive")
    try:
        args.ratios, names = curve_ratios(args.ratios, single_label_mode,
                                         args.prompt_style == "emotion_multilabel_task")
    except ValueError as exc:
        parser.error(str(exc))

    stage_root = args.root.expanduser().resolve()
    model_dir = (args.model_dir or stage_root / "models" / "qwen").expanduser().resolve()
    mosei_root = (args.mosei_root or stage_root / "datasets" / "mosei").expanduser().resolve()
    default_name = (f"qwen2_5_vl_7b_mosei_{method}_curve"
                    + (f"_seed{random_seed}" if method == "random" else "")
                    + ("_single_label_emotion_prompt_cache_success" if single_label_mode and args.decoded_cache_dir
                       else "_single_label_emotion_prompt" if single_label_mode
                       else "_multilabel_emotion_prompt_cache_success"
                       if args.prompt_style == "emotion_multilabel_task" and args.decoded_cache_dir
                       else "_multilabel_emotion_prompt"
                       if args.prompt_style == "emotion_multilabel_task" else ""))
    output_dir = (args.output_dir or stage_root / "results" / default_name).expanduser().resolve()
    protocol = json.loads((Path(__file__).with_name("mosei_video_protocol.json")).read_text())
    protocol["prompt_style"] = args.prompt_style
    ground_truth_csv = (args.single_label_csv or mosei_root / "csv" / "mosei_aligned.csv").expanduser().resolve()
    if not ground_truth_csv.is_file():
        raise FileNotFoundError(ground_truth_csv)
    if single_label_mode:
        single_label_counts(ground_truth_csv)
    try:
        decoded_cache_dir = resolve_cache_frames(args.decoded_cache_dir) if args.decoded_cache_dir else None
    except ValueError as exc:
        parser.error(str(exc))
    if not args.verify_only:
        activate_live_log(output_dir / "run.log")
    environment = verify_qwen_environment(curve=True)
    print(f"Verifying Qwen2.5-VL-7B model files at {model_dir}", flush=True)
    model_info = verify_model(model_dir, "qwen")
    if method in ("fastv", "sparsevlm"):
        model_config = json.loads((model_dir / "config.json").read_text())
        decoder_layers = model_config.get("text_config", model_config)["num_hidden_layers"]
        if method == "fastv" and fastv_k >= decoder_layers:
            parser.error(f"--fastv-k must be smaller than the model's {decoder_layers} decoder layers")
        if method == "sparsevlm" and decoder_layers <= 16:
            parser.error("SparseVLM stages 2, 6, and 15 require at least 17 decoder layers")
    print(f"Verifying MOSEI annotation and clip paths at {mosei_root}", flush=True)
    data_info = verify_mosei(mosei_root, progress_every=500)
    mosei_root = Path(data_info["root"]).resolve()
    annotation_csv = ground_truth_csv
    cache_selection = None
    subset_fields = subset_rows = None
    if decoded_cache_dir:
        selector = cached_single_label_rows if single_label_mode else cached_multilabel_rows
        subset_fields, subset_rows, cache_selection = selector(
            ground_truth_csv, decoded_cache_dir, mosei_root / "Raw", protocol)
        split_counts = cache_selection["selected"]
        annotation_csv = output_dir / ("cached_single_label_subset.csv" if single_label_mode
                                       else "cached_multilabel_subset.csv")
        print(f"Cache-success selection: {json.dumps(cache_selection)}", flush=True)
    else:
        split_counts = single_label_counts(ground_truth_csv) if single_label_mode else SPLIT_COUNTS
    if args.verify_only:
        print(json.dumps({"model": model_info, "data": data_info, "protocol": protocol,
                          "method": method, "random_seed": random_seed,
                          "important_ratio": important_ratio if method == "vispruner" else None,
                          "fastv_k": fastv_k if method == "fastv" else None,
                          "ratios": args.ratios, "seeds": args.seeds,
                          "ground_truth_csv": str(ground_truth_csv),
                          "annotation_csv": str(annotation_csv),
                          "split_counts": split_counts, "cache_selection": cache_selection,
                          "prompt_style": args.prompt_style, "output_dir": str(output_dir),
                          "decoded_cache_dir": str(decoded_cache_dir) if decoded_cache_dir else None,
                          "environment": environment}, indent=2))
        return 0

    import torch
    import transformers
    if transformers.__version__ != "4.57.1":
        raise RuntimeError("Install transformers==4.57.1 before running Qwen2.5 pruning/baseline")
    gpu_name = require_cuda_gpu(torch)

    if subset_rows is not None:
        write_subset(annotation_csv, subset_fields, subset_rows)
    print(f"Starting {method} curve on {gpu_name}; "
          f"ratios={args.ratios} mlp_seeds={args.seeds} random_seed={random_seed} "
          f"important_ratio={important_ratio if method == 'vispruner' else None} "
          f"fastv_k={fastv_k if method == 'fastv' else None} "
          "splits=train,valid,test", flush=True)

    config = {"model_id": MODEL_ID, "model_dir": str(model_dir),
              "model_config_sha256": sha256(model_dir / "config.json"),
              "model_index_sha256": sha256(model_dir / "model.safetensors.index.json"),
              "mosei_root": str(mosei_root),
              "decoded_cache_dir": str(decoded_cache_dir) if decoded_cache_dir else None,
              "ground_truth_csv": str(ground_truth_csv),
              "ground_truth_csv_sha256": sha256(ground_truth_csv),
              "aligned_csv_sha256": sha256(annotation_csv),
              "annotation_csv": str(annotation_csv), "cache_selection": cache_selection,
              "target": ("single highest emotion; positive ties excluded; all-zero neutral"
                         if single_label_mode else "six independent binary emotions; positive iff score > 0"),
              "prompt_style": args.prompt_style,
              "code_revision": git_revision(), "protocol": protocol,
              "measurement_environment": {"gpu": torch.cuda.get_device_name(),
                                          "torch": torch.__version__,
                                          "transformers": transformers.__version__,
                                          "attention": "sdpa", "batch_size": 1,
                                          "flops_profile_target_clips_per_split": FLOP_PROFILE_TARGET},
              "ratios": args.ratios, "seeds": args.seeds, "chunk_size": args.chunk_size,
              "method": {"divprune": "Qwen2.5 DivPrune on projected video tokens",
                         "random": "Qwen2.5 uniform random on projected video tokens",
                         "uniform": "Qwen2.5 temporal-stratified uniform projected video tokens",
                         "vispruner": "Qwen2.5 VisPruner adapted vision-attention and diversity pruning",
                         "visionzip": "Qwen2.5 VisionZip dominant selection and contextual merging, video adapted",
                         "fastv": "Qwen2.5 FastV last-query attention after K decoder layers",
                         "sparsevlm": "Qwen2.5 SparseVLM v1.5-style progressive text-guided pruning with recycling"}[method],
              "test_policy": "All ratios evaluated on test in the same run; validation selects checkpoints"}
    if method == "random":
        config["random_seed"] = random_seed
    if method == "vispruner":
        config["important_ratio"] = important_ratio
        config["attention_source"] = "last_full_qwen_vision_block_mean_received_attention"
    if method == "visionzip":
        config["attention_source"] = "last_full_qwen_vision_block_received_attention_and_keys"
        config["attention_score_reduction"] = "mean_heads_sum_queries_per_attention_segment_v3"
        config["merging"] = "official_qwen_65_to_5_scaled_budget_with_temporal_group_isolation_v1"
    if method == "fastv":
        config["fastv_k"] = fastv_k
        config["attention_source"] = "last_prompt_query_after_k_decoder_layers"
    if method == "sparsevlm":
        config["sparsevlm_stages"] = [2, 6, 15]
        config["sparsevlm_schedule"] = "matrix_rank_lambda_geometric_with_final_budget_cap_v1"
        config["attention_source"] = "selected_post_video_text_queries_at_decoder_stages"
        config["recycling"] = "top_30_percent_discarded_then_density_cluster_mean"
    output_dir.mkdir(parents=True, exist_ok=True)
    config_path = output_dir / "run_config.json"
    record_resume_config(config_path, config)
    env = os.environ.copy()
    env["HF_HUB_OFFLINE"] = "1"
    index_dir = output_dir / "indexes"
    run([sys.executable, str(REPO_ROOT / "scripts" / "data" / "prepare_mosei.py"),
         "--mosei-root", str(mosei_root), "--aligned-csv", str(annotation_csv),
         "--output-dir", str(index_dir),
         *(["--target-mode", "single_max", "--tie-policy", "exclude",
            "--zero-policy", "neutral"] if single_label_mode else [])], env)
    splits = ("train", "valid", "test")
    decode_failures = []
    split_inputs = {}
    work = []
    for ratio, name in zip(args.ratios, names):
        folder = output_dir / name
        split_inputs[name] = {}
        for split in splits:
            expected = split_counts[split]
            profile_stride = max(1, (expected + FLOP_PROFILE_TARGET - 1) // FLOP_PROFILE_TARGET)
            index = index_dir / f"mosei_{split}_index.csv"
            ids = index_ids(index, expected)
            chunks_dir = folder / "chunks" / split
            embedding_pt = folder / "embeddings" / f"mosei_{split}.pt"
            split_inputs[name][split] = {"index": index, "ids": ids,
                                          "chunks_dir": chunks_dir, "embedding_pt": embedding_pt}
            for start in range(0, expected, args.chunk_size):
                end = min(start + args.chunk_size, expected)
                chunk = chunks_dir / f"{split}_{start:05d}_{end:05d}.pt"
                if complete_chunk(chunk, ids[start:end], model_dir, start, ratio,
                                  protocol, profile_stride, method, random_seed, important_ratio,
                                  fastv_k, args.prompt_style):
                    print(f"Reusing complete chunk {chunk}", flush=True)
                    continue
                state, mtime = "new", 0.0
                if chunk.is_file():
                    payload = torch.load(chunk, map_location="cpu", weights_only=False)
                    attempted = len(payload.get("sample_ids", [])) + len(payload.get("errors", []))
                    state = "partial" if attempted < end - start else "retry_errors"
                    mtime = chunk.stat().st_mtime
                work.append({"name": name, "ratio": ratio, "split": split,
                             "index": index, "ids": ids[start:end], "chunk": chunk,
                             "start": start, "end": end, "profile_stride": profile_stride,
                             "state": state, "mtime": mtime, "ordinal": len(work)})

    grouped_work = {}
    for item in work:
        key = (item["split"], item["start"], item["end"])
        grouped_work.setdefault(key, []).append(item)
    # Keep all ratios for a clip chunk together so Decord runs once per clip.
    ordered_groups = sorted(grouped_work.values(), key=lambda group: (
        0 if any(item["state"] == "partial" for item in group) else
        2 if all(item["state"] == "retry_errors" for item in group) else 1,
        -max((item["mtime"] for item in group if item["state"] == "partial"), default=0)
        if any(item["state"] == "partial" for item in group) else
        min(item["ordinal"] for item in group),
    ))
    if ordered_groups and any(item["state"] == "partial" for item in ordered_groups[0]):
        print(f"Resuming partial chunk group first: {ordered_groups[0][0]['split']} "
              f"{ordered_groups[0][0]['start']}:{ordered_groups[0][0]['end']}", flush=True)
    changed_training_conditions = set()
    changed_test_conditions = set()
    for group in ordered_groups:
        active = []
        for item in group:
            name, ratio, split, chunk = item["name"], item["ratio"], item["split"], item["chunk"]
            start, end, profile_stride = item["start"], item["end"], item["profile_stride"]

            def record_decode_failures(payload: dict, item=item) -> None:
                name, ratio, split, chunk = item["name"], item["ratio"], item["split"], item["chunk"]
                start, profile_stride = item["start"], item["profile_stride"]
                chunk_errors = decode_failure_records(
                    payload, item["ids"], model_dir, start, ratio, protocol,
                    profile_stride, method, random_seed, important_ratio, fastv_k,
                    args.prompt_style,
                )
                decode_failures.extend(
                    {"condition": name, "split": split,
                     "sample_id": error["sample_id"], "error": error["error"]}
                    for error in chunk_errors
                )
                print(f"[{name}] Deferring {len(chunk_errors)} saved decoder failures in {chunk}", flush=True)

            if item["state"] == "retry_errors":
                record_decode_failures(torch.load(chunk, map_location="cpu", weights_only=False))
                continue
            active.append(item)
            if split == "test":
                changed_test_conditions.add(name)
            else:
                changed_training_conditions.add(name)
        if not active:
            continue

        first = active[0]
        command = [sys.executable, str(REPO_ROOT / "cluster" / "extract_mosei_qwen2_5vl_multi.py"),
                   "--index-csv", str(first["index"]), "--model-dir", str(model_dir),
                   "--start", str(first["start"]), "--limit", str(first["end"] - first["start"]),
                   "--pruning", method, "--profile-stride", str(first["profile_stride"]),
                   "--prompt-style", args.prompt_style,
                   "--log-every", str(args.log_every), "--save-every", str(args.save_every),
                   "--decode-timeout-seconds", str(args.decode_timeout_seconds)]
        if decoded_cache_dir:
            command.extend(["--decoded-cache-dir", str(decoded_cache_dir)])
        if method == "random":
            command.extend(["--random-seed", str(random_seed)])
        if method == "vispruner":
            command.extend(["--important-ratio", str(important_ratio)])
        if method == "fastv":
            command.extend(["--fastv-k", str(fastv_k)])
        for item in active:
            command.extend(["--ratio-output", str(item["ratio"]), str(item["chunk"])])
        try:
            run(command, env)
        except subprocess.CalledProcessError:
            for item in active:
                if not item["chunk"].is_file():
                    raise
                payload = torch.load(item["chunk"], map_location="cpu", weights_only=False)
                if payload.get("errors"):
                    chunk_errors = decode_failure_records(
                        payload, item["ids"], model_dir, item["start"], item["ratio"],
                        protocol, item["profile_stride"], method, random_seed,
                        important_ratio, fastv_k, args.prompt_style,
                    )
                    decode_failures.extend(
                        {"condition": item["name"], "split": item["split"],
                         "sample_id": error["sample_id"], "error": error["error"]}
                        for error in chunk_errors
                    )
                elif not complete_chunk(item["chunk"], item["ids"], model_dir,
                                        item["start"], item["ratio"], protocol,
                                        item["profile_stride"], method, random_seed,
                                        important_ratio, fastv_k, args.prompt_style):
                    raise

    failures_path = output_dir / "decode_failures.json"
    if decode_failures:
        failures_path.write_text(json.dumps(decode_failures, indent=2) + "\n")
        print(f"Continuing with {len(decode_failures)} recorded decoder failures; "
              f"see {failures_path}", flush=True)
    else:
        failures_path.unlink(missing_ok=True)

    coverage_by_condition = {}
    for ratio, name in zip(args.ratios, names):
        folder = output_dir / name
        retention = {}
        coverage = {}
        for split, info in split_inputs[name].items():
            run([sys.executable, "-m", "scripts", "merge_embedding_chunks",
                 "--chunks-dir", str(info["chunks_dir"]), "--output-pt", str(info["embedding_pt"]),
                 "--pattern", f"{split}_*.pt"], env)
            coverage[split] = verify_available_embeddings(info["embedding_pt"], info["ids"])
            retention[split] = summarize_embedding_cost(
                info["embedding_pt"], ratio, coverage[split]["embedded"])
            retention[split]["indexed_samples"] = coverage[split]["indexed"]
            retention[split]["missing_samples"] = coverage[split]["missing"]
        (folder / "coverage.json").write_text(json.dumps(coverage, indent=2) + "\n")
        (folder / "cost_summary.json").write_text(json.dumps(retention, indent=2) + "\n")
        coverage_by_condition[name] = coverage
        print(f"[{name}] Embeddings merged; split coverage={coverage}", flush=True)

        for seed in args.seeds:
            mlp_dir = folder / "mlp" / f"seed_{seed}"
            metrics_path = mlp_dir / "metrics.json"
            train_changed = (name in changed_training_conditions
                             or not training_metrics_match(metrics_path, coverage, single_label_mode))
            if train_changed:
                classifier = "mosei_single_label_mlp.py" if single_label_mode else "mosei_mlp.py"
                run([sys.executable, str(REPO_ROOT / "scripts" / "classification" / classifier),
                     "train", "--train-pt", str(folder / "embeddings" / "mosei_train.pt"),
                     "--train-index", str(index_dir / "mosei_train_index.csv"),
                     "--valid-pt", str(folder / "embeddings" / "mosei_valid.pt"),
                     "--valid-index", str(index_dir / "mosei_valid_index.csv"),
                     "--output-dir", str(mlp_dir), "--seed", str(seed), "--device", "cuda"], env)
            if not training_metrics_match(metrics_path, coverage, single_label_mode):
                raise RuntimeError(f"Matched MLP metrics do not match extraction coverage: {metrics_path}")
            test_dir = folder / "test_eval" / f"seed_{seed}"
            test_metrics = test_dir / "metrics.json"
            if (train_changed or name in changed_test_conditions
                    or not test_metrics_match(test_metrics, coverage["test"], single_label_mode)):
                classifier = "mosei_single_label_mlp.py" if single_label_mode else "mosei_mlp.py"
                run([sys.executable, str(REPO_ROOT / "scripts" / "classification" / classifier),
                     "evaluate", "--checkpoint", str(mlp_dir / "best_mlp.pt"),
                     "--embeddings-pt", str(folder / "embeddings" / "mosei_test.pt"),
                     "--index", str(index_dir / "mosei_test_index.csv"),
                     "--output-dir", str(test_dir), "--device", "cuda"], env)
            if not test_metrics_match(test_metrics, coverage["test"], single_label_mode):
                raise RuntimeError(f"Test metrics do not match extraction coverage: {test_metrics}")
    write_curve(output_dir, names, tuple(args.seeds), method, random_seed, single_label_mode)
    coverage_complete = all(values["missing"] == 0
                            for condition in coverage_by_condition.values()
                            for values in condition.values())
    (output_dir / "summary.json").write_text(json.dumps({
        "method": method,
        "target": config["target"], "prompt_style": args.prompt_style,
        "annotation_csv": str(annotation_csv), "ground_truth_csv": str(ground_truth_csv),
        "cache_selection": cache_selection,
        "evaluated_splits": splits,
        "coverage_complete": coverage_complete,
        "evaluation_scope": "all_indexed_clips" if coverage_complete else "successfully_decoded_clips_only",
        "coverage_by_condition": coverage_by_condition,
        "decoder_failures": len(decode_failures),
    }, indent=2) + "\n")
    marker = output_dir / ("completed_utc.txt" if coverage_complete else "completed_partial_utc.txt")
    marker.write_text(datetime.now(timezone.utc).isoformat() + "\n")
    (output_dir / ("completed_partial_utc.txt" if coverage_complete else "completed_utc.txt")).unlink(missing_ok=True)
    print(f"{method} curve complete; full coverage={coverage_complete}: "
          f"{output_dir / 'validation_curve.csv'}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
