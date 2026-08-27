#!/usr/bin/env python3
"""Decode MELD videos into cached uint8 frame tensors.

This stage is CPU-heavy and can be run before GPU embedding extraction.
It reads an index CSV from prepare_meld.py and writes one .pt file per sample:

{
    "sample_id": str,
    "frames": torch.uint8 tensor [T, H, W, 3] in RGB order,
    "label": int,
    "label_name": str,
    "metadata": dict,
    "sampling": dict,
}

The saved uint8 frames are not model-normalized. Qwen's processor should do
float conversion and normalization during embedding extraction.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from PIL import Image
from tqdm.auto import tqdm


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Preprocess MELD videos into frame tensors.")
    parser.add_argument("--index-csv", type=Path, required=True, help="CSV from prepare_meld.py.")
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory for per-sample frame .pt files.")
    parser.add_argument("--output-index", type=Path, required=True, help="CSV mapping samples to frame .pt files.")
    parser.add_argument("--fps", type=float, default=6.0, help="Target frame sampling FPS.")
    parser.add_argument("--frame-size", type=int, default=224, help="Square resize size.")
    parser.add_argument("--max-frames", type=int, default=64, help="Maximum sampled frames per video.")
    parser.add_argument("--min-frames", type=int, default=4, help="Minimum sampled frames if video has enough frames.")
    parser.add_argument("--limit", type=int, default=None, help="Optional number of rows to process.")
    parser.add_argument("--start", type=int, default=0, help="Start row offset in index CSV.")
    parser.add_argument("--backend", choices=["decord", "opencv"], default="decord", help="Video decoding backend.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing frame .pt files.")
    return parser.parse_args()


def safe_name(sample_id: str) -> str:
    return "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in sample_id)


def resize_rgb(frame: np.ndarray, frame_size: int) -> np.ndarray:
    image = Image.fromarray(frame, mode="RGB")
    image = image.resize((frame_size, frame_size), Image.BICUBIC)
    return np.asarray(image, dtype=np.uint8)


def choose_indices(total_frames: int, source_fps: float, target_fps: float, min_frames: int, max_frames: int) -> list[int]:
    if total_frames <= 0:
        return []

    if not source_fps or math.isnan(source_fps) or source_fps <= 0:
        source_fps = target_fps

    duration = total_frames / source_fps
    desired = max(1, int(round(duration * target_fps)))
    desired = min(desired, max_frames)
    desired = max(desired, min(min_frames, total_frames, max_frames))
    desired = min(desired, total_frames)

    if desired == 1:
        return [total_frames // 2]

    return np.linspace(0, total_frames - 1, num=desired).round().astype(int).tolist()


def read_video_decord(video_path: Path, fps: float, frame_size: int, min_frames: int, max_frames: int) -> tuple[torch.Tensor, dict[str, Any]]:
    try:
        from decord import VideoReader, cpu
    except ImportError as exc:
        raise ImportError("decord is not installed. Install with: pip install decord") from exc

    vr = VideoReader(str(video_path), ctx=cpu(0))
    total_frames = len(vr)
    source_fps = float(vr.get_avg_fps() or 0.0)
    indices = choose_indices(total_frames, source_fps, fps, min_frames, max_frames)
    if not indices:
        raise ValueError(f"No frames found in {video_path}")

    batch = vr.get_batch(indices).asnumpy()  # RGB uint8 [T,H,W,3]
    resized = np.stack([resize_rgb(frame, frame_size) for frame in batch], axis=0)
    frames = torch.from_numpy(resized.copy()).to(torch.uint8)

    return frames, {
        "backend": "decord",
        "source_fps": source_fps,
        "total_frames": total_frames,
        "sampled_indices": indices,
        "sampled_frames": len(indices),
    }


def read_video_opencv(video_path: Path, fps: float, frame_size: int, min_frames: int, max_frames: int) -> tuple[torch.Tensor, dict[str, Any]]:
    try:
        import cv2
    except ImportError as exc:
        raise ImportError("opencv-python is not installed. Install with: pip install opencv-python") from exc

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    source_fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    indices = choose_indices(total_frames, source_fps, fps, min_frames, max_frames)
    if not indices:
        cap.release()
        raise ValueError(f"No frames found in {video_path}")

    frames = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ok, frame_bgr = cap.read()
        if not ok:
            continue
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        frames.append(resize_rgb(frame_rgb, frame_size))
    cap.release()

    if not frames:
        raise ValueError(f"Could not decode sampled frames from {video_path}")

    tensor = torch.from_numpy(np.stack(frames, axis=0).copy()).to(torch.uint8)
    return tensor, {
        "backend": "opencv",
        "source_fps": source_fps,
        "total_frames": total_frames,
        "sampled_indices": indices,
        "sampled_frames": len(frames),
    }


def decode_video(video_path: Path, backend: str, fps: float, frame_size: int, min_frames: int, max_frames: int) -> tuple[torch.Tensor, dict[str, Any]]:
    if backend == "decord":
        return read_video_decord(video_path, fps, frame_size, min_frames, max_frames)
    if backend == "opencv":
        return read_video_opencv(video_path, fps, frame_size, min_frames, max_frames)
    raise ValueError(f"Unknown backend: {backend}")


def save_frame_payload(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, tmp_path)
    tmp_path.replace(path)


def main() -> None:
    args = parse_args()
    df = pd.read_csv(args.index_csv)

    if "video_exists" in df.columns:
        df = df[df["video_exists"].astype(bool)].copy()
    if args.start:
        df = df.iloc[args.start:].copy()
    if args.limit is not None:
        df = df.head(args.limit).copy()
    df = df.reset_index(drop=True)

    required = ["sample_id", "video_path", "emotion", "emotion_id"]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Index CSV is missing columns: {missing}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.output_index.parent.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    errors: list[dict[str, str]] = []

    print(f"Rows to preprocess: {len(df)}")
    print(f"Backend: {args.backend}")
    print(f"Sampling: fps={args.fps}, frame_size={args.frame_size}, min_frames={args.min_frames}, max_frames={args.max_frames}")
    print(f"Output dir: {args.output_dir}")

    for _, row in tqdm(df.iterrows(), total=len(df)):
        sample_id = str(row["sample_id"])
        video_path = Path(str(row["video_path"]))
        frame_path = args.output_dir / f"{safe_name(sample_id)}.pt"

        output_row = row.to_dict()
        output_row["frames_path"] = str(frame_path.resolve())
        output_row["frames_exists"] = frame_path.exists()
        output_row["num_frames"] = None
        output_row["frame_size"] = args.frame_size
        output_row["sampling_fps"] = args.fps

        try:
            if frame_path.exists() and not args.overwrite:
                payload = torch.load(frame_path, map_location="cpu", weights_only=False)
                frames = payload["frames"]
                output_row["frames_exists"] = True
                output_row["num_frames"] = int(frames.shape[0])
                rows.append(output_row)
                continue

            if not video_path.exists():
                raise FileNotFoundError(str(video_path))

            frames, sampling = decode_video(
                video_path=video_path,
                backend=args.backend,
                fps=args.fps,
                frame_size=args.frame_size,
                min_frames=args.min_frames,
                max_frames=args.max_frames,
            )

            payload = {
                "sample_id": sample_id,
                "frames": frames,  # uint8 [T,H,W,3], RGB
                "label": int(row["emotion_id"]),
                "label_name": str(row["emotion"]),
                "metadata": row.to_dict(),
                "sampling": {
                    **sampling,
                    "target_fps": args.fps,
                    "frame_size": args.frame_size,
                    "min_frames": args.min_frames,
                    "max_frames": args.max_frames,
                    "dtype": "uint8",
                    "layout": "T,H,W,3",
                    "color": "RGB",
                },
            }
            save_frame_payload(frame_path, payload)

            output_row["frames_exists"] = True
            output_row["num_frames"] = int(frames.shape[0])
            rows.append(output_row)
        except Exception as exc:
            output_row["frames_exists"] = False
            rows.append(output_row)
            errors.append({"sample_id": sample_id, "video_path": str(video_path), "error": repr(exc)})

    with args.output_index.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f"Saved frame index: {args.output_index}")
    print(f"Successful frame files: {sum(bool(row['frames_exists']) for row in rows)} / {len(rows)}")
    print(f"Errors: {len(errors)}")

    if errors:
        error_path = args.output_index.with_suffix(".errors.json")
        error_path.write_text(json.dumps(errors, indent=2), encoding="utf-8")
        print(f"Saved errors: {error_path}")


if __name__ == "__main__":
    main()
