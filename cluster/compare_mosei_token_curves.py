#!/usr/bin/env python3
"""Compare matched MOSEI DivPrune, random, uniform, VisPruner, FastV, and SparseVLM curves."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path


def read_run(folder: Path) -> tuple[dict, dict[float, dict], dict]:
    config = json.loads((folder / "run_config.json").read_text())
    rows = json.loads((folder / "validation_curve.json").read_text())
    if not rows:
        raise ValueError(f"No validation-curve rows in {folder}")
    costs = {}
    for row in rows:
        ratio = float(row["requested_retention"])
        if ratio in costs:
            raise ValueError(f"Repeated ratio {ratio} in {folder}")
        costs[ratio] = json.loads((folder / row["condition"] / "cost_summary.json").read_text())["valid"]
    return config, {float(row["requested_retention"]): row for row in rows}, costs


def verify_match(folder: Path, run: tuple[dict, dict[float, dict], dict],
                 div_config: dict, div_rows: dict, div_costs: dict) -> None:
    config, rows, costs = run
    if set(rows) != set(div_rows):
        raise ValueError(f"{folder} does not have the same retention ratios")
    for key in ("model_config_sha256", "model_index_sha256", "aligned_csv_sha256",
                "ground_truth_csv_sha256", "target", "prompt_style", "cache_selection",
                "protocol", "seeds", "measurement_environment"):
        if config.get(key) != div_config.get(key):
            raise ValueError(f"{folder}: {key} differs from the DivPrune run")
    f1_key = ("valid_macro_f1_mean" if div_config.get("prompt_style") == "emotion_task"
              else "valid_macro_f1_at_0.5_mean")
    if any(f1_key not in row for row in rows.values()):
        raise ValueError(f"{folder}: missing {f1_key}")
    for ratio in div_rows:
        if costs[ratio]["profiled_sample_ids"] != div_costs[ratio]["profiled_sample_ids"]:
            raise ValueError(f"{folder}: FLOP-profile sample IDs differ at {ratio}")
        if (costs[ratio]["full_visual_tokens"] != div_costs[ratio]["full_visual_tokens"]
                or costs[ratio]["retained_visual_tokens"] != div_costs[ratio]["retained_visual_tokens"]):
            raise ValueError(f"{folder}: visual-token budget differs at {ratio}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--divprune-dir", type=Path, required=True)
    parser.add_argument("--random-dirs", type=Path, nargs="+",
                        help="One independent random extraction seed per directory")
    parser.add_argument("--uniform-dir", type=Path,
                        help="Deterministic temporal-stratified uniform curve")
    parser.add_argument("--vispruner-dir", type=Path,
                        help="Qwen-adapted VisPruner attention/diversity curve")
    parser.add_argument("--visionzip-dir", type=Path,
                        help="Qwen-adapted VisionZip dominant/contextual curve")
    parser.add_argument("--fastv-dir", type=Path,
                        help="Qwen-adapted FastV in-decoder pruning curve")
    parser.add_argument("--sparsevlm-dir", type=Path,
                        help="Qwen-adapted SparseVLM progressive pruning curve")
    parser.add_argument("--output-csv", type=Path, required=True)
    args = parser.parse_args()
    if not args.random_dirs and not args.uniform_dir and not args.vispruner_dir and not args.visionzip_dir and not args.fastv_dir and not args.sparsevlm_dir:
        parser.error("Supply at least one comparison curve")

    div_config, div_rows, div_costs = read_run(args.divprune_dir)
    single_label_mode = div_config.get("prompt_style") == "emotion_task"
    f1_key = "valid_macro_f1_mean" if single_label_mode else "valid_macro_f1_at_0.5_mean"
    sd_key = "valid_macro_f1_sd" if single_label_mode else "valid_macro_f1_at_0.5_sd"
    if any(f1_key not in row or sd_key not in row for row in div_rows.values()):
        raise ValueError(f"{args.divprune_dir}: missing expected validation metrics for its target")
    random_runs = [read_run(folder) for folder in (args.random_dirs or [])]
    uniform_run = read_run(args.uniform_dir) if args.uniform_dir else None
    vispruner_run = read_run(args.vispruner_dir) if args.vispruner_dir else None
    visionzip_run = read_run(args.visionzip_dir) if args.visionzip_dir else None
    fastv_run = read_run(args.fastv_dir) if args.fastv_dir else None
    sparsevlm_run = read_run(args.sparsevlm_dir) if args.sparsevlm_dir else None
    random_seeds = [config.get("random_seed") for config, _, _ in random_runs]
    if None in random_seeds or len(set(random_seeds)) != len(random_seeds):
        raise ValueError("Random run directories must have distinct extraction seeds")
    if "DivPrune" not in div_config["method"]:
        raise ValueError("--divprune-dir does not contain a DivPrune curve")
    for folder, run in zip(args.random_dirs or [], random_runs):
        config = run[0]
        if "random" not in config["method"]:
            raise ValueError(f"{folder} does not contain a random-token curve")
        verify_match(folder, run, div_config, div_rows, div_costs)
    if uniform_run:
        if "temporal-stratified uniform" not in uniform_run[0]["method"]:
            raise ValueError("--uniform-dir does not contain a uniform-token curve")
        verify_match(args.uniform_dir, uniform_run, div_config, div_rows, div_costs)
    if vispruner_run:
        if "VisPruner" not in vispruner_run[0]["method"]:
            raise ValueError(f"{args.vispruner_dir} does not contain a VisPruner curve")
        verify_match(args.vispruner_dir, vispruner_run, div_config, div_rows, div_costs)
    if visionzip_run:
        if "VisionZip" not in visionzip_run[0]["method"]:
            raise ValueError(f"{args.visionzip_dir} does not contain a VisionZip curve")
        verify_match(args.visionzip_dir, visionzip_run, div_config, div_rows, div_costs)
    if fastv_run:
        if "FastV" not in fastv_run[0]["method"]:
            raise ValueError(f"{args.fastv_dir} does not contain a FastV curve")
        verify_match(args.fastv_dir, fastv_run, div_config, div_rows, div_costs)
    if sparsevlm_run:
        if "SparseVLM" not in sparsevlm_run[0]["method"]:
            raise ValueError(f"{args.sparsevlm_dir} does not contain a SparseVLM curve")
        verify_match(args.sparsevlm_dir, sparsevlm_run, div_config, div_rows, div_costs)

    output_rows = []
    for ratio in sorted(div_rows):
        div = div_rows[ratio]
        result = {
            "target": div_config.get("target"),
            "requested_retention": ratio,
            "actual_retention_valid": div["actual_retention_valid"],
            "mean_retained_visual_tokens_valid": div["mean_retained_visual_tokens_valid"],
            "mean_retained_sequence_tokens_valid": div["mean_retained_sequence_tokens_valid"],
            "divprune_code_revision": div_config.get("code_revision"),
            "divprune_valid_macro_f1_mean": div[f1_key],
            "divprune_mlp_seed_sd": div[sd_key],
            "divprune_valid_mean_end_to_end_seconds": div["valid_mean_end_to_end_seconds"],
            "divprune_valid_peak_allocated_gib": div["valid_peak_total_allocated_gib_mean"],
            "divprune_valid_theoretical_forward_tflops": div["valid_theoretical_forward_tflops_mean"],
        }
        if random_runs:
            random = [rows[ratio] for _, rows, _ in random_runs]
            random_f1 = [float(row[f1_key]) for row in random]
            result.update({
                "random_extraction_seeds": ",".join(str(seed) for seed in random_seeds),
                "random_code_revisions": ",".join(
                    str(config.get("code_revision")) for config, _, _ in random_runs
                ),
                "random_valid_macro_f1_mean": statistics.mean(random_f1),
                "random_extraction_seed_sd": statistics.stdev(random_f1) if len(random_f1) > 1 else 0.0,
                "random_mean_mlp_seed_sd": statistics.mean(
                    float(row[sd_key]) for row in random
                ),
                "divprune_minus_random_macro_f1": (
                    float(div[f1_key]) - statistics.mean(random_f1)
                ),
                "random_valid_mean_end_to_end_seconds": statistics.mean(
                    float(row["valid_mean_end_to_end_seconds"]) for row in random
                ),
                "random_valid_peak_allocated_gib": statistics.mean(
                    float(row["valid_peak_total_allocated_gib_mean"]) for row in random
                ),
                "random_valid_theoretical_forward_tflops": statistics.mean(
                    float(row["valid_theoretical_forward_tflops_mean"]) for row in random
                ),
            })
        if uniform_run:
            uniform = uniform_run[1][ratio]
            result.update({
                "uniform_code_revision": uniform_run[0].get("code_revision"),
                "uniform_valid_macro_f1_mean": uniform[f1_key],
                "uniform_mlp_seed_sd": uniform[sd_key],
                "divprune_minus_uniform_macro_f1": (
                    float(div[f1_key])
                    - float(uniform[f1_key])
                ),
                "uniform_valid_mean_end_to_end_seconds": uniform["valid_mean_end_to_end_seconds"],
                "uniform_valid_peak_allocated_gib": uniform["valid_peak_total_allocated_gib_mean"],
                "uniform_valid_theoretical_forward_tflops": uniform["valid_theoretical_forward_tflops_mean"],
            })
            if random_runs:
                result["uniform_minus_random_macro_f1"] = (
                    float(uniform[f1_key]) - statistics.mean(random_f1)
                )
        if vispruner_run:
            vispruner = vispruner_run[1][ratio]
            result.update({
                "vispruner_code_revision": vispruner_run[0].get("code_revision"),
                "vispruner_important_ratio": vispruner_run[0]["important_ratio"],
                "vispruner_valid_macro_f1_mean": vispruner[f1_key],
                "vispruner_mlp_seed_sd": vispruner[sd_key],
                "vispruner_minus_divprune_macro_f1": (
                    float(vispruner[f1_key])
                    - float(div[f1_key])
                ),
                "vispruner_valid_mean_end_to_end_seconds": vispruner["valid_mean_end_to_end_seconds"],
                "vispruner_valid_peak_allocated_gib": vispruner["valid_peak_total_allocated_gib_mean"],
                "vispruner_valid_theoretical_forward_tflops": vispruner["valid_theoretical_forward_tflops_mean"],
            })
        if visionzip_run:
            visionzip = visionzip_run[1][ratio]
            result.update({
                "visionzip_code_revision": visionzip_run[0].get("code_revision"),
                "visionzip_valid_macro_f1_mean": visionzip[f1_key],
                "visionzip_mlp_seed_sd": visionzip[sd_key],
                "visionzip_minus_divprune_macro_f1": (
                    float(visionzip[f1_key])
                    - float(div[f1_key])
                ),
                "visionzip_valid_mean_end_to_end_seconds": visionzip["valid_mean_end_to_end_seconds"],
                "visionzip_valid_peak_allocated_gib": visionzip["valid_peak_total_allocated_gib_mean"],
                "visionzip_valid_theoretical_forward_tflops": visionzip["valid_theoretical_forward_tflops_mean"],
            })
        if fastv_run:
            fastv = fastv_run[1][ratio]
            result.update({
                "fastv_code_revision": fastv_run[0].get("code_revision"),
                "fastv_k": fastv_run[0]["fastv_k"],
                "fastv_valid_macro_f1_mean": fastv[f1_key],
                "fastv_mlp_seed_sd": fastv[sd_key],
                "fastv_minus_divprune_macro_f1": (
                    float(fastv[f1_key])
                    - float(div[f1_key])
                ),
                "fastv_valid_mean_end_to_end_seconds": fastv["valid_mean_end_to_end_seconds"],
                "fastv_valid_peak_allocated_gib": fastv["valid_peak_total_allocated_gib_mean"],
                "fastv_valid_theoretical_forward_tflops": fastv["valid_theoretical_forward_tflops_mean"],
            })
        if sparsevlm_run:
            sparsevlm = sparsevlm_run[1][ratio]
            result.update({
                "sparsevlm_code_revision": sparsevlm_run[0].get("code_revision"),
                "sparsevlm_valid_macro_f1_mean": sparsevlm[f1_key],
                "sparsevlm_mlp_seed_sd": sparsevlm[sd_key],
                "sparsevlm_minus_divprune_macro_f1": (
                    float(sparsevlm[f1_key])
                    - float(div[f1_key])
                ),
                "sparsevlm_valid_mean_end_to_end_seconds": sparsevlm["valid_mean_end_to_end_seconds"],
                "sparsevlm_valid_peak_allocated_gib": sparsevlm["valid_peak_total_allocated_gib_mean"],
                "sparsevlm_valid_theoretical_forward_tflops": sparsevlm["valid_theoretical_forward_tflops_mean"],
            })
        output_rows.append(result)
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(output_rows[0]))
        writer.writeheader()
        writer.writerows(output_rows)
    print(f"Wrote {len(output_rows)} matched validation-budget rows to {args.output_csv}")
    if any(config.get("code_revision") != div_config.get("code_revision")
           for config, _, _ in random_runs + ([uniform_run] if uniform_run else [])
           + ([vispruner_run] if vispruner_run else [])
           + ([visionzip_run] if visionzip_run else [])
           + ([fastv_run] if fastv_run else [])
           + ([sparsevlm_run] if sparsevlm_run else [])):
        print("Code revisions differ; verify that the shared extraction and MLP paths did not change.")
    if random_runs and len(random_runs) < 3:
        print("For a paper claim, run at least three independent random extraction seeds.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
