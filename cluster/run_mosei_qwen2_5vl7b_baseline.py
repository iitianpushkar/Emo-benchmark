#!/usr/bin/env python3
"""Run the full Qwen2.5-VL-7B frozen-embedding baseline on CMU-MOSEI.

Submit this command inside a single-H100 allocation. It never downloads data or
weights. Completed extraction chunks are reused when the job is restarted.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shlex
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from cluster.live_log import activate_live_log  # noqa: E402
from cluster.preflight import verify_model, verify_mosei  # noqa: E402
from cluster.qwen_cost import FLOP_PROFILE_TARGET, summarize_embedding_cost  # noqa: E402
from cluster.resume_config import record_resume_config  # noqa: E402
from cluster.verify_qwen_environment import require_cuda_gpu, verify_qwen_environment  # noqa: E402

SPLIT_COUNTS = {"train": 16326, "valid": 1871, "test": 4659}
MODEL_ID = "Qwen/Qwen2.5-VL-7B-Instruct"
HIDDEN_SIZE = 3584


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_revision() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def run(command: list[str], env: dict[str, str]) -> None:
    started = time.monotonic()
    print(f"[{datetime.now(timezone.utc).isoformat()}] + {shlex.join(command)}", flush=True)
    child_env = {**env, "PYTHONUNBUFFERED": "1"}
    with subprocess.Popen(command, cwd=REPO_ROOT, env=child_env, text=True,
                          stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                          bufsize=1) as process:
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="", flush=True)
        returncode = process.wait()
    print(f"[{datetime.now(timezone.utc).isoformat()}] Command finished: "
          f"exit={returncode} elapsed={time.monotonic() - started:.1f}s", flush=True)
    if returncode:
        raise subprocess.CalledProcessError(returncode, command)


def index_ids(path: Path, expected: int) -> list[str]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    ids = [row["sample_id"] for row in rows]
    if len(ids) != expected or len(set(ids)) != expected:
        raise ValueError(f"{path}: expected {expected} unique samples, found {len(ids)}")
    if any(row["video_exists"].lower() != "true" for row in rows):
        raise ValueError(f"{path}: some indexed videos are missing")
    return ids


def verify_embeddings(path: Path, expected_ids: list[str]) -> None:
    import torch

    payload = torch.load(path, map_location="cpu", weights_only=False)
    ids = [str(value) for value in payload["sample_ids"]]
    embeddings = payload["embeddings"]
    errors = payload.get("errors", [])
    if errors:
        raise ValueError(f"{path}: {len(errors)} extraction errors; first examples: {errors[:3]}")
    if embeddings.ndim != 2 or embeddings.shape != (len(expected_ids), HIDDEN_SIZE):
        raise ValueError(f"{path}: expected embedding shape {(len(expected_ids), HIDDEN_SIZE)}, got {tuple(embeddings.shape)}")
    if len(set(ids)) != len(ids) or set(ids) != set(expected_ids):
        raise ValueError(f"{path}: sample IDs differ from the split index")
    if not torch.isfinite(embeddings).all():
        raise ValueError(f"{path}: embeddings contain NaN or infinity")
    print(f"Verified {path}: {len(ids)} complete embeddings", flush=True)


def metrics_complete(path: Path) -> bool:
    if not path.is_file() or not (path.parent / "best_mlp.pt").is_file() or not (path.parent / "test_predictions.csv").is_file():
        return False
    try:
        report = json.loads(path.read_text())
        coverage = report["coverage"]
        return (
            all(coverage[split]["embedded"] == expected and coverage[split]["missing"] == 0
                for split, expected in (("train", 16326), ("validation", 1871), ("test", 4659)))
            and report["test"]["sample_count"] == SPLIT_COUNTS["test"]
        )
    except (OSError, ValueError, KeyError, TypeError):
        return False


def complete_baseline_chunk(payload: dict, expected_ids: list[str], model_dir: Path,
                            start: int, video_protocol: dict, profile_stride: int) -> bool:
    saved_ids = payload.get("sample_ids", [])
    config = payload.get("config", {})
    return (
        not payload.get("errors")
        and len(set(saved_ids)) == len(expected_ids)
        and set(saved_ids) == set(expected_ids)
        and tuple(payload["embeddings"].shape) == (len(expected_ids), HIDDEN_SIZE)
        and config.get("model_dir") == str(model_dir)
        and config.get("start") == start
        and config.get("limit") == len(expected_ids)
        and config.get("video_protocol") == video_protocol
        and config.get("retain_ratio") == 1.0
        and config.get("pruning") == "none"
        and config.get("profile_stride") == profile_stride
        and len(payload.get("metadata", [])) == len(expected_ids)
    )


def order_chunk_work(chunks: list[dict]) -> list[dict]:
    """Resume recent partial checkpoints, then new chunks, then old decode failures."""
    return sorted(chunks, key=lambda item: (
        {"partial": 0, "new": 1, "retry_errors": 2}[item["state"]],
        -item["mtime"] if item["state"] == "partial" else item["ordinal"],
    ))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=REPO_ROOT.parents[1],
                        help="Shared parent of codes/, datasets/, models/, and results/")
    parser.add_argument("--model-dir", type=Path, help="Local Qwen2.5-VL-7B-Instruct folder (default: ROOT/models/qwen)")
    parser.add_argument("--mosei-root", type=Path, help="Local MOSEI folder (default: ROOT/datasets/mosei)")
    parser.add_argument("--output-dir", type=Path, help="Run artifacts (default: ROOT/results/qwen2_5_vl_7b_mosei)")
    parser.add_argument("--chunk-size", type=int, default=500)
    parser.add_argument("--log-every", type=int, default=1,
                        help="Print extraction progress every N new clips")
    parser.add_argument("--save-every", type=int, default=25,
                        help="Checkpoint embeddings every N new clips")
    parser.add_argument("--decode-timeout-seconds", type=float, default=10,
                        help="Skip clips whose Decord decode exceeds this many seconds")
    parser.add_argument("--verify-only", action="store_true", help="Validate local inputs without starting GPU work")
    args = parser.parse_args()
    if (args.chunk_size <= 0 or args.log_every <= 0 or args.save_every <= 0
            or args.decode_timeout_seconds <= 0):
        parser.error("--chunk-size, --log-every, --save-every, and --decode-timeout-seconds must be positive")

    root = args.root.expanduser().resolve()
    video_protocol = json.loads((Path(__file__).with_name("mosei_video_protocol.json")).read_text())
    model_dir = (args.model_dir or root / "models" / "qwen").expanduser().resolve()
    mosei_root = (args.mosei_root or root / "datasets" / "mosei").expanduser().resolve()
    output_dir = (args.output_dir or root / "results" / "qwen2_5_vl_7b_mosei").expanduser().resolve()
    environment = verify_qwen_environment()
    print(f"Verifying Qwen2.5-VL-7B model files at {model_dir}", flush=True)
    model = verify_model(model_dir, "qwen")
    print(f"Verifying MOSEI annotation and clip paths at {mosei_root}", flush=True)
    data = verify_mosei(mosei_root, progress_every=500)
    mosei_root = Path(data["root"]).resolve()
    if args.verify_only:
        print(json.dumps({"model_id": MODEL_ID, "model": model, "data": data,
                          "video_protocol": video_protocol, "environment": environment}, indent=2))
        return 0

    import torch
    import transformers
    if transformers.__version__ != "4.57.1":
        raise RuntimeError("Install transformers==4.57.1 before running Qwen2.5 pruning/baseline")

    gpu_name = require_cuda_gpu(torch)

    activate_live_log(output_dir / "run.log")
    print(f"Starting Qwen2.5-VL-7B MOSEI baseline on {gpu_name}", flush=True)

    config = {
        "model_id": MODEL_ID,
        "model_dir": str(model_dir),
        "model_config_sha256": sha256(model_dir / "config.json"),
        "model_index_sha256": sha256(model_dir / "model.safetensors.index.json"),
        "mosei_root": str(mosei_root),
        "aligned_csv_sha256": sha256(mosei_root / "csv" / "mosei_aligned.csv"),
        "code_revision": git_revision(),
        "measurement_environment": {"gpu": torch.cuda.get_device_name(),
                                    "torch": torch.__version__,
                                    "transformers": transformers.__version__,
                                    "attention": "sdpa", "batch_size": 1,
                                    "flops_profile_target_clips_per_split": FLOP_PROFILE_TARGET},
        "chunk_size": args.chunk_size,
        "extraction": {**video_protocol, "modality_mode": "video_text",
                       "pooling": "last", "layer": -1, "save_dtype": "float32"},
        "target": "six independent binary emotions; positive iff score > 0",
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    config_path = output_dir / "run_config.json"
    record_resume_config(config_path, config)

    env = os.environ.copy()
    env["HF_HUB_OFFLINE"] = "1"
    index_dir = output_dir / "indexes"
    run([sys.executable, str(REPO_ROOT / "scripts" / "data" / "prepare_mosei.py"),
         "--mosei-root", str(mosei_root),
         "--output-dir", str(index_dir)], env)

    import torch

    split_inputs = {}
    work = []
    for split, expected in SPLIT_COUNTS.items():
        profile_stride = max(1, (expected + FLOP_PROFILE_TARGET - 1) // FLOP_PROFILE_TARGET)
        index = index_dir / f"mosei_{split}_index.csv"
        ids = index_ids(index, expected)
        split_inputs[split] = {"expected": expected, "ids": ids,
                               "chunks_dir": output_dir / "chunks" / split,
                               "embedding_pt": output_dir / "embeddings" / f"mosei_{split}.pt"}
        for start in range(0, expected, args.chunk_size):
            end = min(start + args.chunk_size, expected)
            chunk = split_inputs[split]["chunks_dir"] / f"{split}_{start:05d}_{end:05d}.pt"
            state, mtime = "new", 0.0
            if chunk.is_file():
                payload = torch.load(chunk, map_location="cpu", weights_only=False)
                if complete_baseline_chunk(payload, ids[start:end], model_dir,
                                           start, video_protocol, profile_stride):
                    print(f"Reusing complete chunk {chunk}", flush=True)
                    continue
                attempted = len(payload.get("sample_ids", [])) + len(payload.get("errors", []))
                state = "partial" if attempted < end - start else "retry_errors"
                mtime = chunk.stat().st_mtime
            work.append({"split": split, "index": index, "ids": ids[start:end],
                         "start": start, "end": end, "chunk": chunk,
                         "profile_stride": profile_stride, "state": state,
                         "mtime": mtime, "ordinal": len(work)})

    ordered_work = order_chunk_work(work)
    if ordered_work and ordered_work[0]["state"] == "partial":
        print(f"Resuming latest partial checkpoint first: {ordered_work[0]['chunk']}", flush=True)
    decode_failures = []
    for item in ordered_work:
        split, index, chunk = item["split"], item["index"], item["chunk"]
        start, end, profile_stride = item["start"], item["end"], item["profile_stride"]
        try:
            run([sys.executable, str(REPO_ROOT / "cluster" / "extract_mosei_qwen2_5vl.py"),
                 "--index-csv", str(index), "--output-pt", str(chunk),
                 "--model-dir", str(model_dir), "--start", str(start),
                 "--limit", str(end - start), "--profile-stride", str(profile_stride),
                 "--log-every", str(args.log_every),
                 "--save-every", str(args.save_every),
                 "--decode-timeout-seconds", str(args.decode_timeout_seconds)], env)
        except subprocess.CalledProcessError:
            if not chunk.is_file():
                raise
            payload = torch.load(chunk, map_location="cpu", weights_only=False)
            chunk_errors = payload.get("errors", [])
            observed_ids = list(payload.get("sample_ids", [])) + [
                error.get("sample_id") for error in chunk_errors
            ]
            chunk_config = payload.get("config", {})
            if (len(observed_ids) != end - start
                    or len(set(observed_ids)) != end - start
                    or set(observed_ids) != set(item["ids"])
                    or chunk_config.get("model_dir") != str(model_dir)
                    or chunk_config.get("start") != start
                    or chunk_config.get("limit") != end - start
                    or chunk_config.get("video_protocol") != video_protocol
                    or chunk_config.get("pruning") != "none"
                    or chunk_config.get("retain_ratio") != 1.0
                    or not chunk_errors
                    or any(not str(error.get("error", "")).startswith(
                        ("TimeoutError: Decord exceeded", "RuntimeError: Video decoding failed"))
                        for error in chunk_errors)):
                raise
            decode_failures.extend(
                {"split": split, "sample_id": error["sample_id"], "error": error["error"]}
                for error in chunk_errors
            )
            print(f"Recorded {len(chunk_errors)} decode failures in {chunk}; "
                  "continuing to the next chunk", flush=True)

    cost_summaries = {}
    failed_splits = {error["split"] for error in decode_failures}
    for split, info in split_inputs.items():
        if split in failed_splits:
            print(f"{split}: decoder failures remain; skipping merge", flush=True)
            continue
        run([sys.executable, "-m", "scripts", "merge_embedding_chunks",
             "--chunks-dir", str(info["chunks_dir"]), "--output-pt", str(info["embedding_pt"]),
             "--pattern", f"{split}_*.pt"], env)
        verify_embeddings(info["embedding_pt"], info["ids"])
        cost_summaries[split] = summarize_embedding_cost(
            info["embedding_pt"], 1.0, info["expected"])
        (output_dir / "cost_summary.json").write_text(json.dumps(cost_summaries, indent=2) + "\n")

    if decode_failures:
        failures_path = output_dir / "decode_failures.json"
        failures_path.write_text(json.dumps(decode_failures, indent=2) + "\n")
        raise RuntimeError(
            f"{len(decode_failures)} clips could not be decoded; all other chunks were saved. "
            f"See {failures_path}. Repair those clips and rerun to finish merging and training."
        )
    (output_dir / "decode_failures.json").unlink(missing_ok=True)

    metrics_path = output_dir / "outputs" / "metrics.json"
    if not metrics_complete(metrics_path):
        run([sys.executable, str(REPO_ROOT / "scripts" / "classification" / "mosei_mlp.py"), "train",
             "--train-pt", str(output_dir / "embeddings" / "mosei_train.pt"),
             "--train-index", str(index_dir / "mosei_train_index.csv"),
             "--valid-pt", str(output_dir / "embeddings" / "mosei_valid.pt"),
             "--valid-index", str(index_dir / "mosei_valid_index.csv"),
             "--test-pt", str(output_dir / "embeddings" / "mosei_test.pt"),
             "--test-index", str(index_dir / "mosei_test_index.csv"),
             "--output-dir", str(output_dir / "outputs"), "--device", "cuda"], env)
    if not metrics_complete(metrics_path):
        raise RuntimeError(f"Classifier metrics incomplete: {metrics_path}")
    (output_dir / "summary.json").write_text(json.dumps({
        "model_id": MODEL_ID,
        "performance": json.loads(metrics_path.read_text()),
        "cost": cost_summaries,
    }, indent=2) + "\n")
    (output_dir / "completed_utc.txt").write_text(datetime.now(timezone.utc).isoformat() + "\n")
    print(f"Full baseline complete: {metrics_path}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
