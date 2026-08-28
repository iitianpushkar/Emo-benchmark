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

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run shared Qwen embedding extraction in resumable chunks.")
    parser.add_argument("--index-csv", type=Path, required=True, help="CSV produced by prepare_meld.py.")
    parser.add_argument("--chunks-dir", type=Path, required=True, help="Directory where chunk .pt files are written.")
    parser.add_argument("--chunk-prefix", required=True, help="Prefix for chunk filenames, e.g. train, dev, test.")
    parser.add_argument("--chunk-size", type=int, default=500, help="Rows per extraction process.")
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
        import torch

        payload = torch.load(path, map_location="cpu", weights_only=False)
        return len(payload.get("sample_ids", [])) + len(payload.get("errors", []))
    except Exception:
        return None


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
