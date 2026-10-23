#!/usr/bin/env python3
"""Run Qwen2.5-VL-7B visual-token retention curves on MOSEI."""

from __future__ import annotations

import argparse
import csv
import json
import os
import statistics
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from cluster.live_log import activate_live_log  # noqa: E402
from cluster.preflight import verify_model, verify_mosei  # noqa: E402
from cluster.qwen_cost import FLOP_PROFILE_TARGET, summarize_embedding_cost  # noqa: E402
from cluster.run_mosei_qwen2_5vl7b_baseline import (  # noqa: E402
    HIDDEN_SIZE, MODEL_ID, SPLIT_COUNTS, git_revision, index_ids, run, sha256,
    verify_embeddings,
)

DEFAULT_RATIOS = (1.0, .9, .8, .7, .6, .5, .4, .3, .25, .2, .15, .1, .05)
DEFAULT_SEEDS = (42, 43, 44)


def ratio_name(ratio: float) -> str:
    percent = round(ratio * 100)
    if abs(ratio * 100 - percent) > 1e-8 or not 5 <= percent <= 100:
        raise ValueError("Ratios must be whole percentages from 5% to 100%")
    return f"r{percent:03d}"


def complete_chunk(path: Path, ids: list[str], model_dir: Path, start: int,
                   ratio: float, protocol: dict, profile_stride: int,
                   method: str = "divprune", random_seed: int | None = None) -> bool:
    if not path.is_file():
        return False
    import torch
    if method == "random":
        from scripts.pruning.random_prune import sample_seed

    payload = torch.load(path, map_location="cpu", weights_only=False)
    saved_ids = [str(value) for value in payload.get("sample_ids", [])]
    config = payload.get("config", {})
    expected_pruning = f"{method}_projected_video_tokens" if ratio < 1 else "none"
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
        and config.get("profile_stride") == profile_stride
        and config.get("video_protocol") == protocol
        and len(payload.get("metadata", [])) == len(ids)
        and (method != "random" or all(
            item.get("random_prune_sample_seed") == sample_seed(random_seed, sample_id)
            for sample_id, item in zip(saved_ids, payload["metadata"])
        ))
    )


def write_curve(root: Path, names: list[str], seeds: tuple[int, ...],
                method: str = "divprune", random_seed: int | None = None) -> None:
    rows = []
    for name in names:
        folder = root / name
        split_summaries = json.loads((folder / "cost_summary.json").read_text())
        reports = [json.loads((folder / "mlp" / f"seed_{seed}" / "metrics.json").read_text())
                   for seed in seeds]
        checkpoint_f1 = [float(report["checkpoint_macro_f1_at_0.5"]) for report in reports]
        tuned_f1 = [float(report["validation"]["macro_f1"]) for report in reports]
        validation_accuracy = [float(report["validation"]["subset_accuracy"]) for report in reports]
        validation_micro_f1 = [float(report["validation"]["micro_f1"]) for report in reports]
        row = {
            "method": method,
            "extraction_seed": random_seed,
            "condition": name,
            "requested_retention": int(name[1:]) / 100,
            "actual_retention_train": split_summaries["train"]["weighted_retention"],
            "actual_retention_valid": split_summaries["valid"]["weighted_retention"],
            "mean_full_visual_tokens_valid": split_summaries["valid"]["mean_full_visual_tokens_per_clip"],
            "mean_retained_visual_tokens_valid": split_summaries["valid"]["mean_retained_visual_tokens_per_clip"],
            "mean_retained_sequence_tokens_valid": split_summaries["valid"]["mean_retained_sequence_tokens_per_clip"],
            "valid_macro_f1_at_0.5_mean": statistics.mean(checkpoint_f1),
            "valid_macro_f1_at_0.5_sd": statistics.stdev(checkpoint_f1) if len(checkpoint_f1) > 1 else 0.0,
            "valid_tuned_macro_f1_mean": statistics.mean(tuned_f1),
            "valid_tuned_macro_f1_sd": statistics.stdev(tuned_f1) if len(tuned_f1) > 1 else 0.0,
            "valid_subset_accuracy_mean": statistics.mean(validation_accuracy),
            "valid_micro_f1_mean": statistics.mean(validation_micro_f1),
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
        test_reports = [folder / "test_eval" / f"seed_{seed}" / "metrics.json" for seed in seeds]
        if all(path.is_file() for path in test_reports):
            test_metrics = [json.loads(path.read_text())["metrics"] for path in test_reports]
            values = [float(metric["macro_f1"]) for metric in test_metrics]
            row["test_macro_f1_mean"] = statistics.mean(values)
            row["test_macro_f1_sd"] = statistics.stdev(values) if len(values) > 1 else 0.0
            row["test_micro_f1_mean"] = statistics.mean(float(metric["micro_f1"]) for metric in test_metrics)
            row["test_subset_accuracy_mean"] = statistics.mean(
                float(metric["subset_accuracy"]) for metric in test_metrics
            )
        rows.append(row)
    rows.sort(key=lambda row: row["requested_retention"])
    baseline = next(row["valid_macro_f1_at_0.5_mean"] for row in rows
                    if row["requested_retention"] == 1)
    for row in rows:
        row["delta_valid_macro_f1_at_0.5"] = row["valid_macro_f1_at_0.5_mean"] - baseline
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
    y = [row["valid_macro_f1_at_0.5_mean"] for row in rows]
    error = [row["valid_macro_f1_at_0.5_sd"] for row in rows]
    axes[0, 0].errorbar(x, y, yerr=error, marker="o", capsize=3)
    axes[0, 0].axhline(baseline, color="gray", linestyle="--", linewidth=1)
    axes[0, 0].set(xlabel="Measured visual-token retention (%)", ylabel="Validation macro F1 at 0.5",
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
    print(f"Wrote {root / 'validation_curve.csv'} and validation_curve.png", flush=True)


def main(method: str = "divprune") -> int:
    if method not in ("divprune", "random"):
        raise ValueError(f"Unknown token pruning method: {method}")
    parser = argparse.ArgumentParser(description=f"Run a Qwen2.5-VL-7B {method} token-retention curve on MOSEI")
    parser.add_argument("--root", type=Path, default=REPO_ROOT.parents[1])
    parser.add_argument("--model-dir", type=Path)
    parser.add_argument("--mosei-root", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--ratios", nargs="+", type=float, default=list(DEFAULT_RATIOS))
    parser.add_argument("--seeds", nargs="+", type=int, default=list(DEFAULT_SEEDS))
    if method == "random":
        parser.add_argument("--random-seed", type=int, default=42,
                            help="Base seed for deterministic per-sample token selection")
    parser.add_argument("--chunk-size", type=int, default=500)
    parser.add_argument("--log-every", type=int, default=1,
                        help="Print extraction progress every N new clips")
    parser.add_argument("--save-every", type=int, default=25,
                        help="Checkpoint embeddings every N new clips")
    parser.add_argument("--verify-only", action="store_true")
    parser.add_argument("--test-all", action="store_true",
                        help="After the validation curve is fixed, evaluate every ratio on test")
    args = parser.parse_args()
    random_seed = args.random_seed if method == "random" else None
    if (args.chunk_size <= 0 or args.log_every <= 0 or args.save_every <= 0 or not args.seeds
            or len(set(args.seeds)) != len(args.seeds)):
        parser.error("Chunk/log/save intervals must be positive and at least one unique seed is required")
    if random_seed is not None and random_seed < 0:
        parser.error("--random-seed must be nonnegative")
    try:
        names = [ratio_name(ratio) for ratio in args.ratios]
    except ValueError as exc:
        parser.error(str(exc))
    if len(names) != len(set(names)) or "r100" not in names:
        parser.error("Ratios must be unique and include 1.0 for the full baseline")

    stage_root = args.root.expanduser().resolve()
    model_dir = (args.model_dir or stage_root / "models" / "qwen").expanduser().resolve()
    mosei_root = (args.mosei_root or stage_root / "datasets" / "mosei").expanduser().resolve()
    default_name = ("qwen2_5_vl_7b_mosei_divprune_curve" if method == "divprune"
                    else f"qwen2_5_vl_7b_mosei_random_curve_seed{random_seed}")
    output_dir = (args.output_dir or stage_root / "results" / default_name).expanduser().resolve()
    protocol = json.loads((Path(__file__).with_name("mosei_video_protocol.json")).read_text())
    model_info = verify_model(model_dir, "qwen")
    data_info = verify_mosei(mosei_root)
    mosei_root = Path(data_info["root"]).resolve()
    if args.verify_only:
        print(json.dumps({"model": model_info, "data": data_info, "protocol": protocol,
                          "method": method, "random_seed": random_seed,
                          "ratios": args.ratios, "seeds": args.seeds}, indent=2))
        return 0

    import torch
    import transformers
    if transformers.__version__ != "4.57.1":
        raise RuntimeError("Install transformers==4.57.1 before running Qwen2.5 pruning/baseline")
    if not torch.cuda.is_available() or not any(
        "H100" in torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())
    ):
        raise RuntimeError("The token-retention curve requires an allocated H100 CUDA GPU")

    activate_live_log(output_dir / "run.log")
    print(f"Starting {method} curve on {torch.cuda.get_device_name()}; "
          f"ratios={args.ratios} mlp_seeds={args.seeds} random_seed={random_seed} "
          f"test_all={args.test_all}", flush=True)

    config = {"model_id": MODEL_ID, "model_dir": str(model_dir),
              "model_config_sha256": sha256(model_dir / "config.json"),
              "model_index_sha256": sha256(model_dir / "model.safetensors.index.json"),
              "mosei_root": str(mosei_root),
              "aligned_csv_sha256": sha256(mosei_root / "csv" / "mosei_aligned.csv"),
              "code_revision": git_revision(), "protocol": protocol,
              "measurement_environment": {"gpu": torch.cuda.get_device_name(),
                                          "torch": torch.__version__,
                                          "transformers": transformers.__version__,
                                          "attention": "sdpa", "batch_size": 1,
                                          "flops_profile_target_clips_per_split": FLOP_PROFILE_TARGET},
              "ratios": args.ratios, "seeds": args.seeds, "chunk_size": args.chunk_size,
              "method": ("Qwen2.5 DivPrune on projected video tokens" if method == "divprune"
                         else "Qwen2.5 uniform random on projected video tokens"),
              "test_policy": "Validation curve first; test-all is optional and cannot select a ratio"}
    if method == "random":
        config["random_seed"] = random_seed
    output_dir.mkdir(parents=True, exist_ok=True)
    config_path = output_dir / "run_config.json"
    if config_path.is_file():
        if json.loads(config_path.read_text()) != config:
            raise ValueError(f"Run configuration changed; use another output directory: {config_path}")
    else:
        config_path.write_text(json.dumps(config, indent=2) + "\n")
    if args.test_all and not (output_dir / "validation_curve.csv").is_file():
        raise ValueError("Run the complete validation curve before --test-all")

    env = os.environ.copy()
    env["HF_HUB_OFFLINE"] = "1"
    index_dir = output_dir / "indexes"
    run([sys.executable, str(REPO_ROOT / "scripts" / "data" / "prepare_mosei.py"),
         "--mosei-root", str(mosei_root), "--output-dir", str(index_dir)], env)
    splits = ("train", "valid", "test") if args.test_all else ("train", "valid")

    for ratio, name in zip(args.ratios, names):
        print(f"[{name}] Starting retention ratio {ratio:.0%}", flush=True)
        folder = output_dir / name
        retention_path = folder / "cost_summary.json"
        retention = json.loads(retention_path.read_text()) if retention_path.is_file() else {}
        for split in splits:
            print(f"[{name}] Starting {split} split ({SPLIT_COUNTS[split]} clips)", flush=True)
            expected = SPLIT_COUNTS[split]
            profile_stride = max(1, (expected + FLOP_PROFILE_TARGET - 1) // FLOP_PROFILE_TARGET)
            index = index_dir / f"mosei_{split}_index.csv"
            ids = index_ids(index, expected)
            chunks_dir = folder / "chunks" / split
            embedding_pt = folder / "embeddings" / f"mosei_{split}.pt"
            for start in range(0, expected, args.chunk_size):
                end = min(start + args.chunk_size, expected)
                chunk = chunks_dir / f"{split}_{start:05d}_{end:05d}.pt"
                if complete_chunk(chunk, ids[start:end], model_dir, start, ratio,
                                  protocol, profile_stride, method, random_seed):
                    print(f"Reusing complete chunk {chunk}", flush=True)
                    continue
                command = [sys.executable, str(REPO_ROOT / "cluster" / "extract_mosei_qwen2_5vl.py"),
                     "--index-csv", str(index), "--output-pt", str(chunk),
                     "--model-dir", str(model_dir), "--start", str(start),
                     "--limit", str(end - start), "--retain-ratio", str(ratio),
                     "--profile-stride", str(profile_stride),
                     "--log-every", str(args.log_every),
                     "--save-every", str(args.save_every)]
                if method == "random":
                    command.extend(["--pruning", "random", "--random-seed", str(random_seed)])
                run(command, env)
            run([sys.executable, "-m", "scripts", "merge_embedding_chunks",
                 "--chunks-dir", str(chunks_dir), "--output-pt", str(embedding_pt),
                 "--pattern", f"{split}_*.pt"], env)
            verify_embeddings(embedding_pt, ids)
            retention[split] = summarize_embedding_cost(embedding_pt, ratio, expected)
        retention_path.write_text(json.dumps(retention, indent=2) + "\n")
        print(f"[{name}] Extraction and cost summaries complete", flush=True)

        for seed in args.seeds:
            mlp_dir = folder / "mlp" / f"seed_{seed}"
            metrics_path = mlp_dir / "metrics.json"
            if not metrics_path.is_file() or not (mlp_dir / "best_mlp.pt").is_file():
                run([sys.executable, str(REPO_ROOT / "scripts" / "classification" / "mosei_mlp.py"),
                     "train", "--train-pt", str(folder / "embeddings" / "mosei_train.pt"),
                     "--train-index", str(index_dir / "mosei_train_index.csv"),
                     "--valid-pt", str(folder / "embeddings" / "mosei_valid.pt"),
                     "--valid-index", str(index_dir / "mosei_valid_index.csv"),
                     "--output-dir", str(mlp_dir), "--seed", str(seed), "--device", "cuda"], env)
            metrics = json.loads(metrics_path.read_text())
            if any(metrics["coverage"][split]["embedded"] != SPLIT_COUNTS[index_split]
                   or metrics["coverage"][split]["missing"] != 0
                   for split, index_split in (("train", "train"), ("validation", "valid"))):
                raise RuntimeError(f"Incomplete matched MLP coverage: {metrics_path}")
            if args.test_all:
                test_dir = folder / "test_eval" / f"seed_{seed}"
                test_metrics = test_dir / "metrics.json"
                if (not test_metrics.is_file()
                        or json.loads(test_metrics.read_text())["metrics"]["sample_count"] != SPLIT_COUNTS["test"]):
                    run([sys.executable, str(REPO_ROOT / "scripts" / "classification" / "mosei_mlp.py"),
                         "evaluate", "--checkpoint", str(mlp_dir / "best_mlp.pt"),
                         "--embeddings-pt", str(folder / "embeddings" / "mosei_test.pt"),
                         "--index", str(index_dir / "mosei_test_index.csv"),
                         "--output-dir", str(test_dir), "--device", "cuda"], env)
    write_curve(output_dir, names, tuple(args.seeds), method, random_seed)
    print(f"{method} curve complete: {output_dir / 'validation_curve.csv'}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
