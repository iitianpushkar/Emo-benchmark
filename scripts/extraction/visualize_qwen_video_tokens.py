#!/usr/bin/env python3
"""Draw Qwen2.5-VL projected video-token ordinals on their source frames.

This only loads the processor configuration; model weights and token scores are
not needed. Run with the same sampling and frame-size settings as extraction.
"""

from __future__ import annotations

# Support both direct execution and `python -m` from the repository root.
import sys as _sys
from pathlib import Path as _Path
if __package__ in (None, ""):
    _sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))

import argparse
import csv
import os
from pathlib import Path


def token_location(ordinal: int, grid: tuple[int, int, int], merge: int) -> tuple[int, int, int]:
    """Return (temporal group, merged-grid row, merged-grid column)."""
    t, h, w = grid
    if merge <= 0 or h % merge or w % merge:
        raise ValueError(f"Invalid spatial merge {merge} for video grid {grid}")
    rows, cols = h // merge, w // merge
    if not 0 <= ordinal < t * rows * cols:
        raise ValueError(f"Token ordinal {ordinal} is outside the video grid")
    group, within_group = divmod(ordinal, rows * cols)
    row, col = divmod(within_group, cols)
    return group, row, col


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("video", type=Path, help="Path to an MP4 or other video supported by qwen-vl-utils")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model-id", default="Qwen/Qwen2.5-VL-3B-Instruct")
    parser.add_argument("--fps", type=float, default=6.0, help="Frame sampling rate used for extraction")
    parser.add_argument("--min-frames", type=int, default=4)
    parser.add_argument("--max-frames", type=int, default=64)
    parser.add_argument("--frame-size", type=int, default=224)
    parser.add_argument("--token", type=int, help="Highlight just this zero-based visual token ordinal")
    return parser.parse_args()


def as_rgb_frame(frame):
    """Convert one qwen-vl-utils sampled frame to an RGB uint8 NumPy array."""
    import numpy as np

    if hasattr(frame, "detach"):
        frame = frame.detach().cpu().numpy()
    elif hasattr(frame, "convert"):
        frame = np.asarray(frame.convert("RGB"))
    else:
        frame = np.asarray(frame)
    if frame.ndim != 3:
        raise ValueError(f"Expected a 3D sampled frame, got shape {frame.shape}")
    if frame.shape[0] in (1, 3) and frame.shape[-1] not in (1, 3):
        frame = np.moveaxis(frame, 0, -1)
    if frame.shape[-1] == 1:
        frame = np.repeat(frame, 3, axis=-1)
    if frame.shape[-1] != 3:
        raise ValueError(f"Expected RGB sampled frame, got shape {frame.shape}")
    if frame.dtype != np.uint8:
        if np.issubdtype(frame.dtype, np.floating) and frame.max() <= 1:
            frame = frame * 255
        frame = np.clip(frame, 0, 255).astype(np.uint8)
    return frame


def main() -> None:
    args = parse_args()
    if not args.video.is_file():
        raise FileNotFoundError(args.video)
    if args.fps <= 0 or args.min_frames <= 0 or args.max_frames < args.min_frames:
        raise ValueError("FPS and frame limits must be positive; max-frames must be >= min-frames")

    # Set before importing qwen-vl-utils, as in the extraction runner.
    os.environ.setdefault("FORCE_QWENVL_VIDEO_READER", "decord")
    os.environ["VIDEO_MAX_PIXELS"] = str(args.frame_size**2 * args.max_frames)
    import cv2
    from qwen_vl_utils import process_vision_info
    from transformers import AutoConfig, AutoProcessor

    from scripts.extraction.extract_qwen_shared_embeddings import fix_video_kwargs, make_message

    messages = make_message(
        str(args.video), "", args.fps, args.min_frames, args.max_frames,
        args.frame_size, "emotion_task", "video_text",
    )
    processor = AutoProcessor.from_pretrained(
        args.model_id, min_pixels=128 * 28 * 28, max_pixels=args.frame_size**2,
    )
    config = AutoConfig.from_pretrained(args.model_id)
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    images, videos, video_kwargs = process_vision_info(messages, return_video_kwargs=True)
    if len(videos) != 1:
        raise ValueError(f"Expected one video, got {len(videos)}")
    inputs = processor(
        text=[text], images=images, videos=videos, padding=True,
        return_tensors="pt", **fix_video_kwargs(video_kwargs),
    )
    grids = inputs["video_grid_thw"].tolist()
    if len(grids) != 1:
        raise ValueError(f"Expected one video grid, got {grids}")
    grid = tuple(map(int, grids[0]))
    vision = config.vision_config
    patch = int(vision.patch_size)
    merge = int(vision.spatial_merge_size)
    temporal_patch = int(vision.temporal_patch_size)
    t, h, w = grid
    count = t * (h // merge) * (w // merge)
    placeholders = int(inputs["input_ids"].eq(config.video_token_id).sum())
    if count != placeholders:
        raise ValueError(f"Grid has {count} tokens but the prompt has {placeholders} video placeholders")
    if args.token is not None:
        token_location(args.token, grid, merge)

    frames = videos[0]
    if len(frames) != t * temporal_patch:
        raise ValueError(
            f"Video grid {grid} and temporal patch size {temporal_patch} imply "
            f"{t * temporal_patch} sampled frames; qwen-vl-utils returned {len(frames)}"
        )
    width, height = w * patch, h * patch
    cell = patch * merge
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_video = args.output_dir / "token_map.mp4"
    writer = cv2.VideoWriter(str(output_video), cv2.VideoWriter_fourcc(*"mp4v"), args.fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"Could not create {output_video}")

    try:
        for frame_index, frame in enumerate(frames):
            rgb = as_rgb_frame(frame)
            # Match the dimensions of the processor's patch grid. The sampled
            # frame may be resized again by AutoProcessor before patchification.
            image = cv2.cvtColor(cv2.resize(rgb, (width, height)), cv2.COLOR_RGB2BGR)
            group = frame_index // temporal_patch
            for row in range(h // merge):
                for col in range(w // merge):
                    ordinal = group * (h // merge) * (w // merge) + row * (w // merge) + col
                    if args.token is not None and ordinal != args.token:
                        continue
                    x0, y0 = col * cell, row * cell
                    color = (0, 255, 255) if args.token is not None else (0, 215, 0)
                    cv2.rectangle(image, (x0, y0), (x0 + cell - 1, y0 + cell - 1), color, 1)
                    label = str(ordinal)
                    font_scale = min(0.45, max(0.25, cell / 85))
                    cv2.putText(image, label, (x0 + 2, y0 + min(cell - 3, 13)),
                                cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), 2, cv2.LINE_AA)
                    cv2.putText(image, label, (x0 + 2, y0 + min(cell - 3, 13)),
                                cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, 1, cv2.LINE_AA)
            if frame_index == 0 or (args.token is not None and group == args.token // ((h // merge) * (w // merge))):
                cv2.imwrite(str(args.output_dir / f"frame_{frame_index:03d}.png"), image)
            writer.write(image)
    finally:
        writer.release()

    with (args.output_dir / "token_map.csv").open("w", newline="") as file:
        csv_writer = csv.writer(file)
        csv_writer.writerow(["token_ordinal", "sampled_frame_start", "sampled_frame_end", "x0", "y0", "x1", "y1"])
        for ordinal in range(count):
            group, row, col = token_location(ordinal, grid, merge)
            csv_writer.writerow([
                ordinal, group * temporal_patch, (group + 1) * temporal_patch - 1,
                col * cell, row * cell, (col + 1) * cell, (row + 1) * cell,
            ])

    print(f"Grid (T,H,W): {grid}; {count} visual tokens; {len(frames)} sampled frames")
    print(f"Video: {output_video}")
    print(f"Token coordinates: {args.output_dir / 'token_map.csv'}")


if __name__ == "__main__":
    main()
