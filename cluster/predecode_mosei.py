#!/usr/bin/env python3
"""Predecode all original CMU-MOSEI clips into a resumable sampled-frame cache."""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
import os
import subprocess
import sys
import threading
import time
from collections import Counter
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from cluster.live_log import activate_live_log  # noqa: E402
from cluster.mosei_decoded_cache import cache_available, cache_paths, identity  # noqa: E402
from cluster.mosei_video_decoder import TimedVideoDecoder  # noqa: E402
from cluster.preflight import verify_mosei  # noqa: E402

SPLIT_COUNTS = {"train": 16326, "valid": 1871, "test": 4659}


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_rows(path: Path, expected: int) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    ids = [row["sample_id"] for row in rows]
    if (len(rows) != expected or len(set(ids)) != expected
            or any(row["video_exists"].lower() != "true" for row in rows)):
        raise ValueError(f"{path}: expected {expected} unique existing clips")
    return rows


def decoder_environment() -> dict[str, str]:
    import torch
    import decord
    from qwen_vl_utils import process_vision_info

    if not callable(process_vision_info):
        raise RuntimeError("qwen-vl-utils process_vision_info is unavailable")
    return {"python": sys.executable, "torch": torch.__version__,
            "decord": importlib.metadata.version("decord"),
            "qwen-vl-utils": importlib.metadata.version("qwen-vl-utils")}


def chunk_state(path: Path, ids: list[str]) -> dict:
    ids_digest = hashlib.sha256("\n".join(ids).encode("utf-8")).hexdigest()
    if path.is_file():
        state = json.loads(path.read_text(encoding="utf-8"))
        if (state.get("ids_sha256") != ids_digest or
                not set(state.get("success_ids", [])) <= set(ids) or
                not set(state.get("failed", {})) <= set(ids) or
                set(state.get("success_ids", [])) & set(state.get("failed", {})) or
                len(state.get("success_ids", [])) != len(set(state.get("success_ids", [])))):
            raise ValueError(f"Chunk state no longer matches source rows: {path}")
    else:
        state = {"ids_sha256": ids_digest, "success_ids": [], "failed": {}}
    return state


def write_manifests(output: Path, cache_dir: Path, expected_counts: dict[str, int]) -> dict:
    successes = {}
    failures = {}
    for split in SPLIT_COUNTS:
        for path in sorted((output / "chunks" / split).glob("*.json")):
            state = json.loads(path.read_text(encoding="utf-8"))
            for sample_id in state["success_ids"]:
                key = (split, sample_id)
                if key in successes or key in failures:
                    raise ValueError(f"Duplicate status for {sample_id}")
                successes[key] = str(cache_paths(cache_dir, sample_id)[0])
            for sample_id, error in state["failed"].items():
                key = (split, sample_id)
                if key in successes or key in failures:
                    raise ValueError(f"Duplicate status for {sample_id}")
                failures[key] = error
    for name, fields, records in (
        ("successful_decodes.csv", ("split", "sample_id", "cache_path"),
         ((*key, path) for key, path in sorted(successes.items()))),
        ("failed_decodes.csv", ("split", "sample_id", "error"),
         ((*key, error) for key, error in sorted(failures.items()))),
    ):
        path = output / name
        temporary = path.with_suffix(".csv.tmp")
        with temporary.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(fields)
            writer.writerows(records)
        temporary.replace(path)
    success_counts = Counter(split for split, _ in successes)
    failure_counts = Counter(split for split, _ in failures)
    summary = {"expected": expected_counts,
               "successful": {split: success_counts[split] for split in expected_counts},
               "failed": {split: failure_counts[split] for split in expected_counts},
               "pending": {split: expected_counts[split] -
                           success_counts[split] - failure_counts[split]
                           for split in expected_counts},
               "cache_dir": str(cache_dir),
               "updated_utc": datetime.now(timezone.utc).isoformat()}
    write_json(output / "summary.json", summary)
    return summary


class ParallelDecoders:
    """Keep one isolated Decord child process per pool thread."""

    def __init__(self, timeout_seconds: float, cache_dir: Path):
        self.timeout_seconds = timeout_seconds
        self.cache_dir = cache_dir
        self.local = threading.local()
        self.lock = threading.Lock()
        self.decoders: list[TimedVideoDecoder] = []

    def decode(self, row: dict, protocol: dict) -> float:
        decoder = getattr(self.local, "decoder", None)
        if decoder is None:
            decoder = TimedVideoDecoder(self.timeout_seconds, cache_dir=self.cache_dir)
            self.local.decoder = decoder
            with self.lock:
                self.decoders.append(decoder)
        video_path = Path(row["video_path"])
        messages = [{"role": "user", "content": [{
            "type": "video", "video": video_path.as_uri(),
            "fps": protocol["fps"], "min_frames": protocol["min_frames"],
            "max_frames": protocol["max_frames"],
            "resized_height": protocol["frame_size"],
            "resized_width": protocol["frame_size"],
        }, {"type": "text", "text": row["utterance"]}]}]
        started = time.monotonic()
        decoder.decode(messages, row["sample_id"], video_path, protocol)
        return time.monotonic() - started

    def close(self) -> None:
        for decoder in self.decoders:
            decoder.close()


def finished_tasks(rows, executor: ThreadPoolExecutor, worker, workers: int):
    """Yield completed rows while limiting queued work to twice the workers."""
    pending = iter(rows)
    futures = {}

    def fill() -> None:
        while len(futures) < 2 * workers:
            try:
                item = next(pending)
            except StopIteration:
                break
            futures[executor.submit(worker, item[1])] = item

    fill()
    try:
        while futures:
            completed, _ = wait(futures, return_when=FIRST_COMPLETED)
            for future in completed:
                yield futures.pop(future), future
            fill()
    finally:
        for future in futures:
            future.cancel()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=REPO_ROOT.parents[1])
    parser.add_argument("--mosei-root", type=Path)
    parser.add_argument("--output-dir", type=Path,
                        help="Default: ROOT/datasets/mosei/decoded_qwen_6fps_224_v1")
    parser.add_argument("--splits", nargs="+", choices=tuple(SPLIT_COUNTS),
                        default=list(SPLIT_COUNTS))
    parser.add_argument("--chunk-size", type=int, default=500)
    parser.add_argument("--workers", type=int, default=1,
                        help="Parallel Decord workers; for 12 CPUs, start with 6")
    parser.add_argument("--save-every", type=int, default=25)
    parser.add_argument("--log-every", type=int, default=1)
    parser.add_argument("--decode-timeout-seconds", type=float, default=10)
    parser.add_argument("--retry-failed", action="store_true")
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    if (args.chunk_size <= 0 or args.workers <= 0 or args.save_every <= 0 or args.log_every <= 0
            or args.decode_timeout_seconds <= 0 or len(set(args.splits)) != len(args.splits)):
        parser.error("Workers, chunk size, save frequency, log frequency, and timeout must be positive; splits must be unique")

    root = args.root.expanduser().resolve()
    mosei = (args.mosei_root or root / "datasets" / "mosei").expanduser().resolve()
    output = (args.output_dir or mosei / "decoded_qwen_6fps_224_v1").expanduser().resolve()
    cache_dir = output / "frames"
    protocol = json.loads((REPO_ROOT / "cluster" / "mosei_video_protocol.json").read_text())
    data = verify_mosei(mosei, progress_every=500)
    environment = decoder_environment()
    print(f"Decoder environment: {environment}", flush=True)
    source_csv = mosei / "csv" / "mosei_aligned.csv"
    config = {"source_csv": str(source_csv), "source_sha256": sha256(source_csv),
              "chunk_size": args.chunk_size,
              "frame_protocol": {key: protocol[key] for key in
                                 ("fps", "min_frames", "max_frames", "frame_size")},
              "decoder": "qwen_vl_utils.process_vision_info/decord",
              "image_patch_size": 14,
              "qwen_vl_utils_version": environment["qwen-vl-utils"]}
    if args.verify_only:
        print(json.dumps({"dataset": data, "output": str(output), "config": config,
                          "environment": environment, "workers": args.workers}, indent=2))
        return 0

    output.mkdir(parents=True, exist_ok=True)
    activate_live_log(output / "run.log")
    config_path = output / "run_config.json"
    if config_path.is_file() and json.loads(config_path.read_text()) != config:
        raise ValueError(f"Cache run configuration changed; use a new --output-dir: {config_path}")
    write_json(config_path, config)
    os.environ["HF_HUB_OFFLINE"] = "1"
    index_dir = output / "indexes"
    subprocess.run([sys.executable, "-u", str(REPO_ROOT / "scripts" / "data" / "prepare_mosei.py"),
                    "--mosei-root", str(mosei), "--output-dir", str(index_dir)],
                   cwd=REPO_ROOT, check=True)
    rows_by_split = {split: load_rows(index_dir / f"mosei_{split}_index.csv", expected)
                     for split, expected in SPLIT_COUNTS.items()}
    print(f"Starting {args.workers} parallel Decord worker(s); "
          f"allocated CPUs={os.environ.get('SLURM_CPUS_PER_TASK', 'unknown')}", flush=True)
    decoders = ParallelDecoders(args.decode_timeout_seconds, cache_dir)
    executor = ThreadPoolExecutor(max_workers=args.workers, thread_name_prefix="mosei_decode")
    try:
        for split in args.splits:
            rows = rows_by_split[split]
            for start in range(0, len(rows), args.chunk_size):
                chunk_rows = rows[start:start + args.chunk_size]
                end = start + len(chunk_rows)
                path = output / "chunks" / split / f"{split}_{start:05d}_{end:05d}.json"
                state = chunk_state(path, [row["sample_id"] for row in chunk_rows])
                success = set(state["success_ids"])
                failed = state["failed"]
                changed = 0
                print(f"[{split}] chunk {start}:{end}; saved={len(success)} failed={len(failed)}", flush=True)
                try:
                    pending_rows = []
                    for offset, row in enumerate(chunk_rows, start=start):
                        sample_id = row["sample_id"]
                        video_path = Path(row["video_path"])
                        expected = identity(sample_id, video_path, protocol)
                        if cache_available(cache_dir, expected):
                            if sample_id not in success or sample_id in failed:
                                success.add(sample_id)
                                failed.pop(sample_id, None)
                                changed += 1
                            continue
                        # A source-video change or removed cache invalidates a prior success.
                        success.discard(sample_id)
                        if sample_id in failed and not args.retry_failed:
                            continue
                        pending_rows.append((offset, row))
                    if changed:
                        state["success_ids"] = sorted(success)
                        write_json(path, state)
                    chunk_started = time.monotonic()
                    completed_new = 0
                    for (offset, row), future in finished_tasks(
                            pending_rows, executor, lambda item: decoders.decode(item, protocol),
                            args.workers):
                        sample_id = row["sample_id"]
                        try:
                            duration = future.result()
                            success.add(sample_id)
                            failed.pop(sample_id, None)
                            print(f"[{split}] {offset + 1}/{len(rows)} {sample_id}: decoded "
                                  f"in {duration:.1f}s", flush=True)
                        except Exception as exc:
                            failed[sample_id] = f"{type(exc).__name__}: {exc}"
                            print(f"[{split}] {offset + 1}/{len(rows)} {sample_id}: "
                                  f"FAILED {failed[sample_id]}", flush=True)
                        changed += 1
                        completed_new += 1
                        if changed % args.save_every == 0 or sample_id in failed:
                            state["success_ids"] = sorted(success)
                            write_json(path, state)
                        if changed % args.log_every == 0:
                            elapsed = max(time.monotonic() - chunk_started, 1e-9)
                            rate = completed_new / elapsed
                            eta = (len(pending_rows) - completed_new) / rate if rate else 0
                            print(f"[{split}] chunk progress {len(success) + len(failed)}/{len(chunk_rows)}; "
                                  f"success={len(success)} failed={len(failed)} "
                                  f"rate={rate:.2f} clips/s ETA={eta:.0f}s", flush=True)
                finally:
                    state["success_ids"] = sorted(success)
                    write_json(path, state)
                summary = write_manifests(output, cache_dir, SPLIT_COUNTS)
                print(f"Checkpointed {path}; coverage={summary}", flush=True)
    finally:
        executor.shutdown(wait=True, cancel_futures=True)
        decoders.close()
        write_manifests(output, cache_dir, SPLIT_COUNTS)
    print(f"Predecode complete: {output / 'summary.json'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
