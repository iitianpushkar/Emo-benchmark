#!/usr/bin/env python3
"""Run SmolVLM2 embedding extraction in separate resumable processes."""

from __future__ import annotations

import argparse
import math
import subprocess
import sys
from pathlib import Path

import pandas as pd
import torch
import transformers

DEFAULT_BENCHMARK_FPS = 6.0
OFFICIAL_FPS = 1.0
OFFICIAL_MAX_FRAMES = 64

EXPECTED_CONFIG_KEYS = (
    "model_id",
    "fps",
    "max_frames",
    "skip_secs",
    "pooling",
    "prompt_style",
    "no_generation_prompt",
    "modality_mode",
    "compute_dtype",
    "layer",
)


def chunk_path(chunks_dir: Path, prefix: str, start: int, end: int) -> Path:
    return chunks_dir / f"{prefix}_{start:05d}_{end:05d}.pt"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run shared SmolVLM2 extraction in resumable chunks.")
    parser.add_argument("--index-csv", type=Path, required=True)
    parser.add_argument("--chunks-dir", type=Path, required=True)
    parser.add_argument("--chunk-prefix", required=True, help="Usually train, dev, or test.")
    parser.add_argument("--chunk-size", type=int, default=500)
    parser.add_argument("--model-id", default="HuggingFaceTB/SmolVLM2-2.2B-Instruct")
    parser.add_argument(
        "--fps",
        type=float,
        default=DEFAULT_BENCHMARK_FPS,
        help="Benchmark default: 6 FPS. Pass 1 for the official checkpoint default.",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=OFFICIAL_MAX_FRAMES,
        help="Official checkpoint default: 64.",
    )
    parser.add_argument("--skip-secs", type=float, default=1.0)
    parser.add_argument("--pooling", choices=["last", "mean", "max"], default="last")
    parser.add_argument("--prompt-style", choices=["emotion_task", "utterance_only"], default="emotion_task")
    parser.add_argument(
        "--modality-mode",
        choices=["video_text", "video_only", "text_only"],
        default="video_text",
    )
    parser.add_argument("--save-dtype", choices=["float16", "float32"], default="float32")
    parser.add_argument("--compute-dtype", choices=["auto", "float16", "bfloat16"], default="auto")
    parser.add_argument("--gc-every", type=int, default=5)
    parser.add_argument("--batch-save-every", type=int, default=25)
    parser.add_argument("--layer", type=int, default=-1)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--resume", action="store_true", default=True)
    parser.add_argument("--no-resume", action="store_false", dest="resume")
    parser.add_argument("--retry-errors", action="store_true")
    parser.add_argument("--skip-existing-complete", action="store_true")
    parser.add_argument("--no-generation-prompt", action="store_true")
    return parser.parse_args()


def existing_chunk_count(path: Path, args: argparse.Namespace) -> int | None:
    if not path.exists():
        return None
    payload = torch.load(path, map_location="cpu", weights_only=False)
    config = payload.get("config", {})
    for key in EXPECTED_CONFIG_KEYS:
        if key not in config:
            raise ValueError(f"Cannot reuse {path}; stored config is missing {key}")
        if config[key] != getattr(args, key):
            raise ValueError(
                f"Cannot reuse {path}; {key} changed from {config[key]!r} to {getattr(args, key)!r}"
            )
    if args.compute_dtype == "bfloat16":
        resolved_dtype = torch.bfloat16
    elif args.compute_dtype == "float16":
        resolved_dtype = torch.float16
    elif torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        resolved_dtype = torch.bfloat16
    else:
        resolved_dtype = torch.float16
    expected_runtime = {
        "resolved_compute_dtype": str(resolved_dtype),
        "transformers_version": transformers.__version__,
    }
    for key, value in expected_runtime.items():
        if config.get(key) != value:
            raise ValueError(
                f"Cannot reuse {path}; {key} changed from {config.get(key)!r} to {value!r}"
            )
    return len(payload.get("sample_ids", [])) + len(payload.get("errors", []))


def main() -> None:
    args = parse_args()
    if args.chunk_size <= 0:
        raise ValueError("--chunk-size must be positive")

    extractor = Path(__file__).resolve().parent / "extract_smolvlm2_embeddings.py"
    df = pd.read_csv(args.index_csv)
    if args.modality_mode != "text_only" and "video_exists" in df.columns:
        df = df[df["video_exists"].astype(bool)].copy()
    total_available = len(df)

    start = args.start
    stop = total_available if args.limit is None else min(total_available, start + args.limit)
    if start < 0 or start >= total_available:
        raise ValueError(f"--start {start} is outside available rows: {total_available}")
    if stop <= start:
        raise ValueError(f"No rows selected: start={start}, stop={stop}")

    args.chunks_dir.mkdir(parents=True, exist_ok=True)
    print(f"Index rows available: {total_available}")
    print(f"Chunk range: [{start}, {stop})")
    print(f"Chunk size: {args.chunk_size}; chunks: {math.ceil((stop - start) / args.chunk_size)}")
    print(f"Chunks dir: {args.chunks_dir}")
    print(f"Extractor: {extractor}")
    print("Spatial processing: official checkpoint-native 384x384 (not user-overridden)")
    if args.fps != OFFICIAL_FPS or args.max_frames != OFFICIAL_MAX_FRAMES:
        print("WARNING: temporal sampling differs from official defaults (1 FPS, max 64 frames).")

    for chunk_start in range(start, stop, args.chunk_size):
        chunk_end = min(chunk_start + args.chunk_size, stop)
        current_limit = chunk_end - chunk_start
        out_path = chunk_path(args.chunks_dir, args.chunk_prefix, chunk_start, chunk_end)

        if args.skip_existing_complete:
            count = existing_chunk_count(out_path, args)
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
            "--skip-secs",
            str(args.skip_secs),
            "--pooling",
            args.pooling,
            "--prompt-style",
            args.prompt_style,
            "--modality-mode",
            args.modality_mode,
            "--save-dtype",
            args.save_dtype,
            "--compute-dtype",
            args.compute_dtype,
            "--gc-every",
            str(args.gc_every),
            "--batch-save-every",
            str(args.batch_save_every),
            "--layer",
            str(args.layer),
        ]
        if args.resume:
            cmd.append("--resume")
        if args.retry_errors:
            cmd.append("--retry-errors")
        if args.no_generation_prompt:
            cmd.append("--no-generation-prompt")

        print(f"\n=== Running chunk {chunk_start}:{chunk_end} -> {out_path} ===")
        result = subprocess.run(cmd, check=False)
        if result.returncode != 0:
            raise RuntimeError(f"Chunk {chunk_start}:{chunk_end} failed with exit code {result.returncode}")

    print("All requested chunks finished.")


if __name__ == "__main__":
    main()
