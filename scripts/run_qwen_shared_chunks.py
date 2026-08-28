#!/usr/bin/env python3
"""Run Qwen shared embedding extraction in separate resumable chunks.

Each chunk launches extract_qwen_shared_embeddings.py as a separate Python
process. This is intentional: when a chunk process exits, CPU RAM used by
video decoding/model processing is released by the OS.
"""

from __future__ import annotations

import argparse
import math
import subprocess
import sys
from pathlib import Path
from typing import Any

import pandas as pd
import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run shared Qwen embedding extraction in resumable chunks.")
    parser.add_argument("--index-csv", type=Path, required=True, help="CSV produced by prepare_meld.py.")
    parser.add_argument("--chunks-dir", type=Path, required=True, help="Directory where chunk .pt files are written.")
    parser.add_argument("--chunk-prefix", required=True, help="Prefix for chunk filenames, e.g. train, dev, test.")
    parser.add_argument("--chunk-size", type=int, default=500, help="Rows per extraction process.")
    parser.add_argument(
        "--seed-pt",
        type=Path,
        default=None,
        help="Existing merged/partial .pt file. Its sample_ids are copied into matching chunk files before extraction.",
    )
    parser.add_argument("--extractor", type=Path, default=None, help="Path to extract_qwen_shared_embeddings.py.")
    parser.add_argument("--model-id", default="Qwen/Qwen2.5-VL-3B-Instruct", help="Hugging Face model id.")
    parser.add_argument("--fps", type=float, default=6.0, help="Video sampling FPS.")
    parser.add_argument("--max-frames", type=int, default=64, help="Maximum sampled frames per video.")
    parser.add_argument("--min-frames", type=int, default=4, help="Minimum sampled frames per video.")
    parser.add_argument("--frame-size", type=int, default=224, help="Square resize size for video frames.")
    parser.add_argument("--pooling", choices=["last", "mean", "max"], default="last", help="Pooling used by extractor.")
    parser.add_argument("--prompt-style", choices=["emotion_task", "utterance_only"], default="emotion_task")
    parser.add_argument("--save-dtype", choices=["float16", "float32"], default="float32")
    parser.add_argument("--gc-every", type=int, default=5)
    parser.add_argument("--batch-save-every", type=int, default=25)
    parser.add_argument("--layer", type=int, default=-1)
    parser.add_argument("--video-max-pixels", type=int, default=None)
    parser.add_argument("--start", type=int, default=0, help="Global start row in index CSV.")
    parser.add_argument("--limit", type=int, default=None, help="Optional total rows to cover from --start.")
    parser.add_argument("--resume", action="store_true", default=True, help="Resume existing chunk files. Enabled by default.")
    parser.add_argument("--no-resume", action="store_false", dest="resume", help="Overwrite/recompute chunk files.")
    parser.add_argument("--retry-errors", action="store_true", help="Retry samples stored as errors inside a chunk.")
    parser.add_argument("--skip-existing-complete", action="store_true", help="Skip chunk files that already contain the expected number of rows/errors.")
    parser.add_argument("--skip-decord-precheck", action="store_true")
    parser.add_argument("--no-generation-prompt", action="store_true")
    parser.add_argument("--trust-remote-code", action="store_true")
    return parser.parse_args()


def default_extractor_path() -> Path:
    return Path(__file__).resolve().parent / "extract_qwen_shared_embeddings.py"


def chunk_path(chunks_dir: Path, prefix: str, start: int, end: int) -> Path:
    return chunks_dir / f"{prefix}_{start:05d}_{end:05d}.pt"


def existing_chunk_count(path: Path) -> int | None:
    if not path.exists():
        return None
    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
        return len(payload.get("sample_ids", [])) + len(payload.get("errors", []))
    except Exception:
        return None


def save_payload(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, tmp_path)
    tmp_path.replace(path)


def seed_chunks_from_existing_pt(
    seed_pt: Path,
    df: pd.DataFrame,
    chunks_dir: Path,
    prefix: str,
    chunk_size: int,
    start: int,
    stop: int,
) -> None:
    if seed_pt is None or not seed_pt.exists():
        return

    payload = torch.load(seed_pt, map_location="cpu", weights_only=False)
    required = ["embeddings", "labels", "sample_ids"]
    missing = [key for key in required if key not in payload]
    if missing:
        raise ValueError(f"Cannot seed chunks from {seed_pt}; missing keys: {missing}")

    sample_ids = [str(item) for item in payload.get("sample_ids", [])]
    embeddings = payload["embeddings"].cpu()
    labels = payload["labels"].cpu().long()
    metadata = list(payload.get("metadata", [{} for _ in sample_ids]))
    errors = list(payload.get("errors", []))

    if embeddings.shape[0] != len(sample_ids) or labels.shape[0] != len(sample_ids):
        raise ValueError(f"Cannot seed chunks from {seed_pt}; embeddings/labels/sample_ids lengths differ")

    id_to_index = {str(row.sample_id): int(pos) for pos, row in df.reset_index(drop=True).iterrows()}
    buckets: dict[tuple[int, int], dict[str, Any]] = {}

    for i, sample_id in enumerate(sample_ids):
        global_index = id_to_index.get(sample_id)
        if global_index is None or global_index < start or global_index >= stop:
            continue
        chunk_start = start + ((global_index - start) // chunk_size) * chunk_size
        chunk_end = min(chunk_start + chunk_size, stop)
        bucket = buckets.setdefault(
            (chunk_start, chunk_end),
            {
                "embeddings": [],
                "labels": [],
                "sample_ids": [],
                "metadata": [],
                "errors": [],
            },
        )
        bucket["embeddings"].append(embeddings[i])
        bucket["labels"].append(labels[i])
        bucket["sample_ids"].append(sample_id)
        bucket["metadata"].append(metadata[i] if i < len(metadata) else {})

    for error in errors:
        sample_id = str(error.get("sample_id", ""))
        global_index = id_to_index.get(sample_id)
        if global_index is None or global_index < start or global_index >= stop:
            continue
        chunk_start = start + ((global_index - start) // chunk_size) * chunk_size
        chunk_end = min(chunk_start + chunk_size, stop)
        bucket = buckets.setdefault(
            (chunk_start, chunk_end),
            {
                "embeddings": [],
                "labels": [],
                "sample_ids": [],
                "metadata": [],
                "errors": [],
            },
        )
        bucket["errors"].append(error)

    for (chunk_start, chunk_end), bucket in sorted(buckets.items()):
        out_path = chunk_path(chunks_dir, prefix, chunk_start, chunk_end)
        if out_path.exists():
            print(f"Seed skipped existing chunk: {out_path}")
            continue
        if bucket["embeddings"]:
            chunk_embeddings = torch.stack(bucket["embeddings"])
            chunk_labels = torch.stack(bucket["labels"]).long()
        else:
            chunk_embeddings = embeddings.new_empty((0, embeddings.shape[1]))
            chunk_labels = labels.new_empty((0,), dtype=torch.long)

        chunk_payload = {
            "embeddings": chunk_embeddings,
            "labels": chunk_labels,
            "label_names": payload.get("label_names"),
            "sample_ids": bucket["sample_ids"],
            "metadata": bucket["metadata"],
            "errors": bucket["errors"],
            "config": {
                "seeded_from": str(seed_pt),
                "chunk_start": chunk_start,
                "chunk_end": chunk_end,
                "embedding_type": payload.get("config", {}).get("embedding_type", "shared_video_text"),
            },
        }
        save_payload(out_path, chunk_payload)
        print(
            f"Seeded {out_path}: "
            f"{len(bucket['sample_ids'])} embeddings, {len(bucket['errors'])} errors"
        )


def main() -> None:
    args = parse_args()
    if args.chunk_size <= 0:
        raise ValueError("--chunk-size must be positive")

    extractor = args.extractor or default_extractor_path()
    if not extractor.exists():
        raise FileNotFoundError(f"Extractor not found: {extractor}")

    df = pd.read_csv(args.index_csv)
    if "video_exists" in df.columns:
        df = df[df["video_exists"].astype(bool)].copy()
    total_available = len(df)

    start = args.start
    stop = total_available if args.limit is None else min(total_available, start + args.limit)
    if start < 0 or start >= total_available:
        raise ValueError(f"--start {start} is outside available rows: {total_available}")
    if stop <= start:
        raise ValueError(f"No rows selected: start={start}, stop={stop}")

    args.chunks_dir.mkdir(parents=True, exist_ok=True)
    total_chunks = math.ceil((stop - start) / args.chunk_size)

    print(f"Index rows available: {total_available}")
    print(f"Chunk range: [{start}, {stop})")
    print(f"Chunk size: {args.chunk_size}; chunks: {total_chunks}")
    print(f"Chunks dir: {args.chunks_dir}")
    print(f"Extractor: {extractor}")
    if args.seed_pt is not None:
        print(f"Seed file: {args.seed_pt}")
        seed_chunks_from_existing_pt(
            seed_pt=args.seed_pt,
            df=df,
            chunks_dir=args.chunks_dir,
            prefix=args.chunk_prefix,
            chunk_size=args.chunk_size,
            start=start,
            stop=stop,
        )

    for chunk_start in range(start, stop, args.chunk_size):
        chunk_end = min(chunk_start + args.chunk_size, stop)
        current_limit = chunk_end - chunk_start
        out_path = chunk_path(args.chunks_dir, args.chunk_prefix, chunk_start, chunk_end)

        if args.skip_existing_complete:
            count = existing_chunk_count(out_path)
            if count is not None and count >= current_limit:
                print(f"Skipping complete chunk {out_path} ({count}/{current_limit})")
                continue

        cmd = [
            sys.executable,
            str(extractor),
            "--index-csv",
            str(args.index_csv),
            "--output-pt",
            str(out_path),
            "--start",
            str(chunk_start),
            "--limit",
            str(current_limit),
            "--model-id",
            args.model_id,
            "--fps",
            str(args.fps),
            "--max-frames",
            str(args.max_frames),
            "--min-frames",
            str(args.min_frames),
            "--frame-size",
            str(args.frame_size),
            "--pooling",
            args.pooling,
            "--prompt-style",
            args.prompt_style,
            "--save-dtype",
            args.save_dtype,
            "--gc-every",
            str(args.gc_every),
            "--batch-save-every",
            str(args.batch_save_every),
            "--layer",
            str(args.layer),
        ]

        if args.video_max_pixels is not None:
            cmd.extend(["--video-max-pixels", str(args.video_max_pixels)])
        if args.resume:
            cmd.append("--resume")
        if args.retry_errors:
            cmd.append("--retry-errors")
        if args.skip_decord_precheck:
            cmd.append("--skip-decord-precheck")
        if args.no_generation_prompt:
            cmd.append("--no-generation-prompt")
        if args.trust_remote_code:
            cmd.append("--trust-remote-code")

        print(f"\n=== Running chunk {chunk_start}:{chunk_end} -> {out_path} ===")
        result = subprocess.run(cmd, check=False)
        if result.returncode != 0:
            raise RuntimeError(f"Chunk {chunk_start}:{chunk_end} failed with exit code {result.returncode}")

    print("All requested chunks finished.")


if __name__ == "__main__":
    main()
