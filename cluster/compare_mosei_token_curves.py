#!/usr/bin/env python3
"""Compare matched MOSEI DivPrune and random-token validation curves."""

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


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--divprune-dir", type=Path, required=True)
    parser.add_argument("--random-dirs", type=Path, nargs="+", required=True,
                        help="One independent random extraction seed per directory")
    parser.add_argument("--output-csv", type=Path, required=True)
    args = parser.parse_args()

    div_config, div_rows, div_costs = read_run(args.divprune_dir)
    random_runs = [read_run(folder) for folder in args.random_dirs]
    random_seeds = [config.get("random_seed") for config, _, _ in random_runs]
    if None in random_seeds or len(set(random_seeds)) != len(random_seeds):
        raise ValueError("Random run directories must have distinct extraction seeds")
    if "DivPrune" not in div_config["method"]:
        raise ValueError("--divprune-dir does not contain a DivPrune curve")
    ratios = set(div_rows)
    for folder, (config, rows, costs) in zip(args.random_dirs, random_runs):
        if "random" not in config["method"]:
            raise ValueError(f"{folder} does not contain a random-token curve")
        if set(rows) != ratios:
            raise ValueError(f"{folder} does not have the same retention ratios")
        for key in ("model_config_sha256", "model_index_sha256", "aligned_csv_sha256",
                    "protocol", "seeds", "measurement_environment"):
            if config.get(key) != div_config.get(key):
                raise ValueError(f"{folder}: {key} differs from the DivPrune run")
        for ratio in ratios:
            if costs[ratio]["profiled_sample_ids"] != div_costs[ratio]["profiled_sample_ids"]:
                raise ValueError(f"{folder}: FLOP-profile sample IDs differ at {ratio}")
            if (costs[ratio]["full_visual_tokens"] != div_costs[ratio]["full_visual_tokens"]
                    or costs[ratio]["retained_visual_tokens"] != div_costs[ratio]["retained_visual_tokens"]):
                raise ValueError(f"{folder}: visual-token budget differs at {ratio}")

    output_rows = []
    for ratio in sorted(ratios):
        div = div_rows[ratio]
        random = [rows[ratio] for _, rows, _ in random_runs]
        random_f1 = [float(row["valid_macro_f1_at_0.5_mean"]) for row in random]
        output_rows.append({
            "requested_retention": ratio,
            "actual_retention_valid": div["actual_retention_valid"],
            "random_extraction_seeds": ",".join(str(seed) for seed in random_seeds),
            "divprune_code_revision": div_config.get("code_revision"),
            "random_code_revisions": ",".join(
                str(config.get("code_revision")) for config, _, _ in random_runs
            ),
            "divprune_valid_macro_f1_mean": div["valid_macro_f1_at_0.5_mean"],
            "divprune_mlp_seed_sd": div["valid_macro_f1_at_0.5_sd"],
            "random_valid_macro_f1_mean": statistics.mean(random_f1),
            "random_extraction_seed_sd": statistics.stdev(random_f1) if len(random_f1) > 1 else 0.0,
            "random_mean_mlp_seed_sd": statistics.mean(
                float(row["valid_macro_f1_at_0.5_sd"]) for row in random
            ),
            "divprune_minus_random_macro_f1": (
                float(div["valid_macro_f1_at_0.5_mean"]) - statistics.mean(random_f1)
            ),
            "divprune_valid_mean_end_to_end_seconds": div["valid_mean_end_to_end_seconds"],
            "random_valid_mean_end_to_end_seconds": statistics.mean(
                float(row["valid_mean_end_to_end_seconds"]) for row in random
            ),
            "divprune_valid_theoretical_forward_tflops": div["valid_theoretical_forward_tflops_mean"],
            "random_valid_theoretical_forward_tflops": statistics.mean(
                float(row["valid_theoretical_forward_tflops_mean"]) for row in random
            ),
        })
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(output_rows[0]))
        writer.writeheader()
        writer.writerows(output_rows)
    print(f"Wrote {len(output_rows)} paired validation-budget rows to {args.output_csv}")
    if any(config.get("code_revision") != div_config.get("code_revision")
           for config, _, _ in random_runs):
        print("Code revisions differ; verify that the shared extraction and MLP paths did not change.")
    if len(random_runs) < 3:
        print("For a paper claim, run at least three independent random extraction seeds.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
