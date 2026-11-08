#!/usr/bin/env python3
"""Run cached-frame Qwen2.5-VL-7B MELD baseline or pruning curves."""

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

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from cluster.live_log import activate_live_log  # noqa: E402
from cluster.meld_cache_subset import (SPLITS, attach_labels, selected_cached_rows,
                                       write_selected_indexes)  # noqa: E402
from cluster.mosei_cache_subset import resolve_cache_frames  # noqa: E402
from cluster.preflight import verify_meld, verify_model  # noqa: E402
from cluster.qwen_cost import FLOP_PROFILE_TARGET, summarize_embedding_cost  # noqa: E402
from cluster.resume_config import record_resume_config  # noqa: E402
from cluster.run_mosei_divprune_curve import (DEFAULT_SEEDS, complete_chunk,
                                              decode_failure_records, ratio_name)  # noqa: E402
from cluster.run_mosei_qwen2_5vl7b_baseline import (HIDDEN_SIZE, MODEL_ID, git_revision,
                                                    index_ids, run, sha256,
                                                    verify_available_embeddings)  # noqa: E402
from cluster.verify_qwen_environment import require_cuda_gpu, verify_qwen_environment  # noqa: E402

RATIOS = (.05, .1, .15, .2, .3, .5, .7)
METHODS = ("divprune", "random", "uniform", "vispruner", "visionzip", "fastv", "sparsevlm")


def metrics_complete(folder: Path, seed: int, ids_by_split: dict[str, list[str]]) -> bool:
    path = folder / "metrics.json"
    if not path.is_file() or not (folder / "best_mlp.pt").is_file():
        return False
    try:
        report = json.loads(path.read_text())
        if not all(key in report[split] for split in ("best_dev", "test")
                   for key in ("macro_f1", "accuracy", "weighted_f1")):
            return False
        for split, filename in (("dev", "dev_predictions.csv"), ("test", "test_predictions.csv")):
            with (folder / filename).open(newline="", encoding="utf-8") as handle:
                saved_ids = [row["sample_id"] for row in csv.DictReader(handle)]
            if len(saved_ids) != len(ids_by_split[split]) or set(saved_ids) != set(ids_by_split[split]):
                return False
        import torch
        checkpoint = torch.load(folder / "best_mlp.pt", map_location="cpu", weights_only=False)
        return (checkpoint["input_dim"] == HIDDEN_SIZE and
                checkpoint["args"]["seed"] == seed and
                checkpoint["label_names"] == ["anger", "disgust", "fear", "joy",
                                              "neutral", "sadness", "surprise"])
    except (OSError, ValueError, KeyError, TypeError, RuntimeError, EOFError):
        return False


def write_curve(output: Path, names: list[str], seeds: list[int], method: str) -> list[dict]:
    rows = []
    for name in names:
        folder = output / name
        coverage = json.loads((folder / "coverage.json").read_text())
        cost = json.loads((folder / "cost_summary.json").read_text())
        reports = [json.loads((folder / "mlp" / f"seed_{seed}" / "metrics.json").read_text())
                   for seed in seeds]
        row = {"method": method, "condition": name,
               "requested_retention": int(name[1:]) / 100,
               "actual_retention_dev": cost["dev"]["weighted_retention"],
               "actual_retention_test": cost["test"]["weighted_retention"],
               "mean_retained_visual_tokens_dev": cost["dev"]["mean_retained_visual_tokens_per_clip"],
               "dev_mean_end_to_end_seconds": cost["dev"]["latency_seconds_per_clip"]["end_to_end_seconds"]["mean"],
               "dev_peak_total_allocated_gib_mean": cost["dev"]["peak_total_allocated_gib"]["mean"],
               "dev_theoretical_forward_tflops_mean": cost["dev"]["theoretical_forward_tflops_per_profiled_clip"]["mean"]}
        for split in SPLITS:
            for key in ("indexed", "embedded", "missing"):
                row[f"{split}_{key}"] = coverage[split][key]
        for split, report_key in (("dev", "best_dev"), ("test", "test")):
            for metric in ("macro_f1", "accuracy", "weighted_f1"):
                values = [float(report[report_key][metric]) for report in reports]
                row[f"{split}_{metric}_mean"] = statistics.mean(values)
                row[f"{split}_{metric}_sd"] = statistics.stdev(values) if len(values) > 1 else 0.0
        rows.append(row)
    rows.sort(key=lambda row: row["requested_retention"])
    with (output / "validation_curve.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    (output / "validation_curve.json").write_text(json.dumps(rows, indent=2) + "\n")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    x = [row["requested_retention"] * 100 for row in rows]
    figure, axes = plt.subplots(2, 2, figsize=(11, 8))
    for axis, split, metric, title in (
        (axes[0, 0], "test", "macro_f1", "Test macro F1"),
        (axes[0, 1], "test", "accuracy", "Test accuracy"),
        (axes[1, 0], "test", "weighted_f1", "Test weighted F1"),
        (axes[1, 1], "dev", "macro_f1", "Dev macro F1"),
    ):
        axis.errorbar(x, [row[f"{split}_{metric}_mean"] for row in rows],
                      yerr=[row[f"{split}_{metric}_sd"] for row in rows],
                      marker="o", capsize=3)
        axis.set(xlabel="Requested visual-token retention (%)", ylabel=title,
                 title="Mean ± SD across MLP seeds")
        axis.grid(alpha=0.25)
    figure.tight_layout()
    figure.savefig(output / "test_metrics_curve.png", dpi=180)
    plt.close(figure)
    return rows


def main(method: str | None = None) -> int:
    if method is not None and method not in METHODS:
        raise ValueError(f"Unknown pruning method: {method}")
    baseline = method is None
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=REPO.parents[1])
    parser.add_argument("--meld-root", type=Path)
    parser.add_argument("--model-dir", type=Path)
    parser.add_argument("--decoded-cache-dir", type=Path)
    parser.add_argument("--output-dir", type=Path)
    if not baseline:
        parser.add_argument("--ratios", nargs="+", type=float)
    parser.add_argument("--seeds", nargs="+", type=int, default=list(DEFAULT_SEEDS))
    parser.add_argument("--chunk-size", type=int, default=500)
    parser.add_argument("--save-every", type=int, default=25)
    parser.add_argument("--log-every", type=int, default=1)
    parser.add_argument("--decode-timeout-seconds", type=float, default=10)
    parser.add_argument("--verify-only", action="store_true")
    if method == "random":
        parser.add_argument("--random-seed", type=int, default=42)
    if method == "vispruner":
        parser.add_argument("--important-ratio", type=float, default=0.5)
    if method == "fastv":
        parser.add_argument("--fastv-k", type=int, default=2)
    args = parser.parse_args()
    if (args.chunk_size <= 0 or args.save_every <= 0 or args.log_every <= 0
            or args.decode_timeout_seconds <= 0 or not args.seeds
            or len(args.seeds) != len(set(args.seeds))):
        parser.error("Chunk/log/save/timeout values must be positive and MLP seeds unique")
    ratios = [1.0] if baseline else list(args.ratios if args.ratios is not None else RATIOS)
    names = [ratio_name(ratio) for ratio in ratios]
    if len(set(names)) != len(names) or (not baseline and "r100" in names):
        parser.error("Pruning ratios must be unique; reserve 100% for the full baseline")
    random_seed = args.random_seed if method == "random" else None
    important_ratio = args.important_ratio if method == "vispruner" else 0.5
    fastv_k = args.fastv_k if method == "fastv" else 2
    if random_seed is not None and random_seed < 0:
        parser.error("--random-seed must be nonnegative")
    if not 0 <= important_ratio <= 1 or fastv_k < 1:
        parser.error("Invalid pruning settings")

    root = args.root.expanduser().resolve()
    meld = (args.meld_root or root / "datasets" / "meld").expanduser().resolve()
    model_dir = (args.model_dir or root / "models" / "qwen").expanduser().resolve()
    cache_root = (args.decoded_cache_dir or meld / "decoded_qwen_6fps_224_v1").expanduser().resolve()
    cache_dir = resolve_cache_frames(cache_root)
    suffix = ("qwen2_5_vl_7b_meld_emotion_prompt_cache_success" if baseline else
              f"qwen2_5_vl_7b_meld_{method}_curve"
              + (f"_seed{random_seed}" if method == "random" else "")
              + "_emotion_prompt_cache_success")
    output = (args.output_dir or root / "results" / suffix).expanduser().resolve()
    if not args.verify_only:
        activate_live_log(output / "run.log")
    environment = verify_qwen_environment(curve=True)
    model_info = verify_model(model_dir, "qwen")
    meld_info = verify_meld(meld)
    meld = Path(meld_info["root"]).resolve()
    protocol_path = Path(__file__).with_name("meld_video_protocol.json")
    protocol = json.loads(protocol_path.read_text())
    protocol["prompt_style"] = "meld_emotion_task"
    env = os.environ.copy()
    env["HF_HUB_OFFLINE"] = "1"
    all_indexes = output / "indexes_all"
    run([sys.executable, "-u", str(REPO / "scripts" / "data" / "prepare_meld.py"),
         "--meld-root", str(meld), "--output-dir", str(all_indexes)], env)
    rows_by_split, cache_selection = selected_cached_rows(all_indexes, cache_dir, protocol)
    counts = cache_selection["selected"]
    print(f"MELD cache-success train/dev/test counts: {counts}; "
          f"stale entries={cache_selection['stale_success_entries']}", flush=True)
    if args.verify_only:
        print(json.dumps({"model": model_info, "meld": meld_info,
                          "environment": environment, "cache_selection": cache_selection,
                          "ratios": ratios, "seeds": args.seeds, "output_dir": str(output)}, indent=2))
        return 0

    import torch
    import transformers
    if transformers.__version__ != "4.57.1":
        raise RuntimeError("Install transformers==4.57.1 for Qwen2.5-VL-7B")
    gpu = require_cuda_gpu(torch)
    if method in ("fastv", "sparsevlm"):
        model_config = json.loads((model_dir / "config.json").read_text())
        layers = model_config.get("text_config", model_config)["num_hidden_layers"]
        if method == "fastv" and fastv_k >= layers:
            parser.error(f"--fastv-k must be below {layers}")
        if method == "sparsevlm" and layers <= 16:
            parser.error("SparseVLM requires at least 17 decoder layers")
    print(f"Starting MELD {'full baseline' if baseline else method} on {gpu}; "
          f"ratios={ratios} MLP seeds={args.seeds}", flush=True)
    output.mkdir(parents=True, exist_ok=True)
    index_dir = output / "indexes"
    write_selected_indexes(index_dir, rows_by_split)
    config = {"model_id": MODEL_ID, "model_dir": str(model_dir),
              "model_config_sha256": sha256(model_dir / "config.json"),
              "model_index_sha256": sha256(model_dir / "model.safetensors.index.json"),
              "meld_root": str(meld), "cache_dir": str(cache_dir),
              "cache_selection": cache_selection,
              "cache_identity_sha256": cache_selection["cache_identity_sha256"],
              "selected_csv_sha256": sha256(output / "cached_meld_subset.csv"),
              "protocol": protocol, "prompt_style": "meld_emotion_task",
              "target": "seven mutually exclusive MELD emotions",
              "method": method or "full", "ratios": ratios, "seeds": args.seeds,
              "random_seed": random_seed, "important_ratio": important_ratio,
              "fastv_k": fastv_k, "chunk_size": args.chunk_size,
              "code_revision": git_revision(), "gpu": gpu,
              "test_policy": "Evaluate test in the same run after dev-selected MLP checkpoint"}
    config_path = output / "run_config.json"
    if config_path.is_file():
        prior = json.loads(config_path.read_text())
        critical = ("model_config_sha256", "model_index_sha256", "selected_csv_sha256",
                    "protocol", "cache_dir", "cache_identity_sha256")
        changed_inputs = [key for key in critical if prior.get(key) != config[key]]
        if changed_inputs:
            raise ValueError(f"Cached MELD inputs changed ({changed_inputs}); use a new --output-dir")
    record_resume_config(config_path, config)

    ids_by_split = {split: index_ids(index_dir / f"meld_{split}_index.csv", counts[split])
                    for split in SPLITS}
    work = []
    split_inputs = {}
    for ratio, name in zip(ratios, names):
        split_inputs[name] = {}
        folder = output / name
        for split in SPLITS:
            ids = ids_by_split[split]
            profile_stride = max(1, (len(ids) + FLOP_PROFILE_TARGET - 1) // FLOP_PROFILE_TARGET)
            index = index_dir / f"meld_{split}_index.csv"
            chunks = folder / "chunks" / split
            embedding = folder / "embeddings" / f"meld_{split}.pt"
            split_inputs[name][split] = (index, ids, chunks, embedding)
            for start in range(0, len(ids), args.chunk_size):
                end = min(start + args.chunk_size, len(ids))
                chunk = chunks / f"{split}_{start:05d}_{end:05d}.pt"
                if complete_chunk(chunk, ids[start:end], model_dir, start, ratio,
                                  protocol, profile_stride, method or "divprune",
                                  random_seed, important_ratio, fastv_k, "meld_emotion_task"):
                    continue
                state = "new"
                if chunk.is_file():
                    payload = torch.load(chunk, map_location="cpu", weights_only=False)
                    attempted = len(payload.get("sample_ids", [])) + len(payload.get("errors", []))
                    state = "partial" if attempted < end - start else "retry_errors"
                work.append({"name": name, "ratio": ratio, "split": split,
                             "index": index, "ids": ids[start:end], "chunk": chunk,
                             "start": start, "end": end, "profile_stride": profile_stride,
                             "state": state})

    grouped = {}
    for item in work:
        grouped.setdefault((item["split"], item["start"], item["end"]), []).append(item)
    changed = set()
    failures = []
    for group in grouped.values():
        active = []
        for item in group:
            if item["state"] == "retry_errors":
                payload = torch.load(item["chunk"], map_location="cpu", weights_only=False)
                errors = decode_failure_records(payload, item["ids"], model_dir,
                                                item["start"], item["ratio"], protocol,
                                                item["profile_stride"], method or "divprune",
                                                random_seed, important_ratio, fastv_k,
                                                "meld_emotion_task")
                failures.extend({"condition": item["name"], "split": item["split"],
                                 **error} for error in errors)
            else:
                active.append(item)
                changed.add(item["name"])
        if not active:
            continue
        first = active[0]
        command = [sys.executable, "-u", str(REPO / "cluster" / "extract_mosei_qwen2_5vl_multi.py"),
                   "--index-csv", str(first["index"]), "--model-dir", str(model_dir),
                   "--start", str(first["start"]), "--limit", str(first["end"] - first["start"]),
                   "--pruning", method or "divprune", "--prompt-style", "meld_emotion_task",
                   "--video-protocol-json", str(protocol_path),
                   "--decoded-cache-dir", str(cache_dir),
                   "--profile-stride", str(first["profile_stride"]),
                   "--save-every", str(args.save_every), "--log-every", str(args.log_every),
                   "--decode-timeout-seconds", str(args.decode_timeout_seconds)]
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
                    errors = decode_failure_records(payload, item["ids"], model_dir,
                                                    item["start"], item["ratio"], protocol,
                                                    item["profile_stride"], method or "divprune",
                                                    random_seed, important_ratio, fastv_k,
                                                    "meld_emotion_task")
                    failures.extend({"condition": item["name"], "split": item["split"],
                                     **error} for error in errors)
                elif not complete_chunk(item["chunk"], item["ids"], model_dir,
                                        item["start"], item["ratio"], protocol,
                                        item["profile_stride"], method or "divprune",
                                        random_seed, important_ratio, fastv_k,
                                        "meld_emotion_task"):
                    raise
    if failures:
        (output / "decode_failures.json").write_text(json.dumps(failures, indent=2) + "\n")
    else:
        (output / "decode_failures.json").unlink(missing_ok=True)

    coverage_by_condition = {}
    for ratio, name in zip(ratios, names):
        folder = output / name
        coverage = {}
        cost = {}
        embedded_ids = {}
        for split in SPLITS:
            index, ids, chunks, embedding = split_inputs[name][split]
            run([sys.executable, "-m", "scripts", "merge_embedding_chunks",
                 "--chunks-dir", str(chunks), "--output-pt", str(embedding),
                 "--pattern", f"{split}_*.pt"], env)
            coverage[split] = verify_available_embeddings(embedding, ids)
            attach_labels(embedding, index)
            payload = torch.load(embedding, map_location="cpu", weights_only=False)
            embedded_ids[split] = [str(value) for value in payload["sample_ids"]]
            cost[split] = summarize_embedding_cost(embedding, ratio, coverage[split]["embedded"])
        (folder / "coverage.json").write_text(json.dumps(coverage, indent=2) + "\n")
        (folder / "cost_summary.json").write_text(json.dumps(cost, indent=2) + "\n")
        coverage_by_condition[name] = coverage
        for seed in args.seeds:
            mlp_dir = folder / "mlp" / f"seed_{seed}"
            if name in changed or not metrics_complete(mlp_dir, seed, embedded_ids):
                run([sys.executable, "-u", str(REPO / "scripts" / "classification" / "train_mlp.py"),
                     "--train-pt", str(folder / "embeddings" / "meld_train.pt"),
                     "--dev-pt", str(folder / "embeddings" / "meld_dev.pt"),
                     "--test-pt", str(folder / "embeddings" / "meld_test.pt"),
                     "--output-dir", str(mlp_dir), "--seed", str(seed),
                     "--device", "cuda"], env)
            if not metrics_complete(mlp_dir, seed, embedded_ids):
                raise RuntimeError(f"MLP metrics do not match extracted MELD clips: {mlp_dir}")
    rows = write_curve(output, names, args.seeds, method or "full")
    complete = all(c["missing"] == 0 for values in coverage_by_condition.values()
                   for c in values.values())
    (output / "summary.json").write_text(json.dumps({
        "method": method or "full", "model_id": MODEL_ID,
        "target": config["target"], "prompt_style": "meld_emotion_task",
        "cache_selection": cache_selection, "coverage_by_condition": coverage_by_condition,
        "coverage_complete": complete, "decoder_failures": len(failures),
        "evaluation_scope": "all_selected_cached_clips" if complete else "successfully_loaded_cached_clips_only",
        "metrics_by_ratio": rows}, indent=2) + "\n")
    marker = output / ("completed_utc.txt" if complete else "completed_partial_utc.txt")
    marker.write_text(datetime.now(timezone.utc).isoformat() + "\n")
    print(f"MELD run complete: {output / 'test_metrics_curve.png'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
