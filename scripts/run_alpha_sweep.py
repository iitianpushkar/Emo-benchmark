#!/usr/bin/env python3
"""Run alpha-weighted modality fusion and MLP training across many alpha values."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


DEFAULT_ALPHAS = "0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run alpha sweep for video/text LM-space embedding fusion.")
    parser.add_argument("--video-dir", type=Path, required=True, help="Directory containing video_only meld_*.pt files.")
    parser.add_argument("--text-dir", type=Path, required=True, help="Directory containing text_only meld_*.pt files.")
    parser.add_argument("--fused-root", type=Path, required=True, help="Root directory for fused alpha embeddings.")
    parser.add_argument("--output-root", type=Path, required=True, help="Root directory for trained alpha MLP outputs.")
    parser.add_argument("--alphas", default=DEFAULT_ALPHAS, help="Comma-separated alpha/video weights.")
    parser.add_argument("--hidden-dim", type=int, default=512)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"], default="auto")
    parser.add_argument("--no-normalize", action="store_true", help="Pass through to fuse_modality_embeddings.py.")
    parser.add_argument("--no-test", action="store_true", help="Do not require or train/evaluate on test embeddings.")
    return parser.parse_args()


def parse_alphas(value: str) -> list[float]:
    alphas = [float(item.strip()) for item in value.split(",") if item.strip()]
    if not alphas:
        raise ValueError("At least one alpha is required")
    for alpha in alphas:
        if not 0.0 <= alpha <= 1.0:
            raise ValueError(f"Alpha must be between 0 and 1, got {alpha}")
    return alphas


def alpha_dir_name(alpha: float) -> str:
    return f"alpha_{alpha:.2f}".replace(".", "_")


def run_command(command: list[str]) -> None:
    print(" ".join(command))
    subprocess.run(command, check=True)


def main() -> None:
    args = parse_args()
    alphas = parse_alphas(args.alphas)

    video_train = args.video_dir / "meld_train.pt"
    video_dev = args.video_dir / "meld_dev.pt"
    video_test = args.video_dir / "meld_test.pt"
    text_train = args.text_dir / "meld_train.pt"
    text_dev = args.text_dir / "meld_dev.pt"
    text_test = args.text_dir / "meld_test.pt"

    required = [video_train, video_dev, text_train, text_dev]
    if not args.no_test:
        required.extend([video_test, text_test])
    missing = [path for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing required embedding files: {[str(path) for path in missing]}")

    script_dir = Path(__file__).resolve().parent
    fuse_script = script_dir / "fuse_modality_embeddings.py"
    train_script = script_dir / "train_mlp.py"

    for alpha in alphas:
        run_name = alpha_dir_name(alpha)
        fused_dir = args.fused_root / run_name
        output_dir = args.output_root / run_name

        fuse_cmd = [
            sys.executable,
            str(fuse_script),
            "--video-train-pt",
            str(video_train),
            "--text-train-pt",
            str(text_train),
            "--video-dev-pt",
            str(video_dev),
            "--text-dev-pt",
            str(text_dev),
            "--output-dir",
            str(fused_dir),
            "--alpha",
            str(alpha),
        ]
        if not args.no_test:
            fuse_cmd.extend(["--video-test-pt", str(video_test), "--text-test-pt", str(text_test)])
        if args.no_normalize:
            fuse_cmd.append("--no-normalize")
        run_command(fuse_cmd)

        train_cmd = [
            sys.executable,
            str(train_script),
            "--train-pt",
            str(fused_dir / "meld_train.pt"),
            "--dev-pt",
            str(fused_dir / "meld_dev.pt"),
            "--output-dir",
            str(output_dir),
            "--hidden-dim",
            str(args.hidden_dim),
            "--dropout",
            str(args.dropout),
            "--epochs",
            str(args.epochs),
            "--batch-size",
            str(args.batch_size),
            "--learning-rate",
            str(args.learning_rate),
            "--weight-decay",
            str(args.weight_decay),
            "--seed",
            str(args.seed),
            "--device",
            args.device,
        ]
        if not args.no_test:
            train_cmd.extend(["--test-pt", str(fused_dir / "meld_test.pt")])
        run_command(train_cmd)


if __name__ == "__main__":
    main()
