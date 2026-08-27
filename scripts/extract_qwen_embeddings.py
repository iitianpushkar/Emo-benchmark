#!/usr/bin/env python3
"""Extract frozen Qwen2.5-VL video embeddings for MELD.

This script follows the mentor-requested setup:
  video -> sample frames -> Qwen visual encoder/projector -> pooled embedding

It does not call model.generate(). The default embedding is taken from
Qwen2.5-VL's get_video_features(...) path, i.e. before language decoding.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
from pathlib import Path
from typing import Any

# These must be set before importing qwen_vl_utils.
os.environ.setdefault("FORCE_QWENVL_VIDEO_READER", "decord")

import pandas as pd
import torch
from qwen_vl_utils import process_vision_info
from tqdm.auto import tqdm
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

LABELS = ["anger", "disgust", "fear", "joy", "neutral", "sadness", "surprise"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract Qwen2.5-VL embeddings from MELD videos.")
    parser.add_argument("--index-csv", type=Path, required=True, help="CSV produced by prepare_meld.py.")
    parser.add_argument("--output-pt", type=Path, required=True, help="Path to save extracted embeddings .pt file.")
    parser.add_argument("--model-id", default="Qwen/Qwen2.5-VL-3B-Instruct", help="Hugging Face model id.")
    parser.add_argument("--fps", type=float, default=6.0, help="Video sampling FPS.")
    parser.add_argument("--max-frames", type=int, default=64, help="Maximum sampled frames per video.")
    parser.add_argument("--min-frames", type=int, default=4, help="Minimum sampled frames per video.")
    parser.add_argument("--frame-size", type=int, default=224, help="Square resize size for video frames.")
    parser.add_argument(
        "--video-max-pixels",
        type=int,
        default=None,
        help="Optional VIDEO_MAX_PIXELS env cap. Defaults to frame_size * frame_size * max_frames.",
    )
    parser.add_argument("--limit", type=int, default=None, help="Optional number of rows to process.")
    parser.add_argument("--start", type=int, default=0, help="Start row offset in index CSV.")
    parser.add_argument("--batch-save-every", type=int, default=25, help="Save checkpoint after this many samples.")
    parser.add_argument(
        "--pooling",
        choices=["mean", "max"],
        default="mean",
        help="How to pool token/frame embeddings into one vector per sample.",
    )
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Pass trust_remote_code=True to from_pretrained if needed by a model variant.",
    )
    return parser.parse_args()


def make_message(video_path: str, fps: float, min_frames: int, max_frames: int, frame_size: int) -> list[dict[str, Any]]:
    return [
        {
            "role": "user",
            "content": [
                {
                    "type": "video",
                    "video": video_path,
                    "fps": fps,
                    "min_frames": min_frames,
                    "max_frames": max_frames,
                    "resized_height": frame_size,
                    "resized_width": frame_size,
                },
                # Minimal text is included so the processor builds a valid multimodal input.
                # The embedding extraction below uses the video feature path, not generation.
                {"type": "text", "text": "Extract video representation."},
            ],
        }
    ]


def fix_video_kwargs(video_kwargs: dict[str, Any]) -> dict[str, Any]:
    fixed = dict(video_kwargs)
    # Compatibility fix for qwen-vl-utils / transformers combinations.
    if "fps" in fixed and isinstance(fixed["fps"], list) and len(fixed["fps"]) == 1:
        fixed["fps"] = fixed["fps"][0]
    return fixed


def pool_tokens(x: torch.Tensor, mode: str) -> torch.Tensor:
    """Pool [tokens, dim] or [batch, tokens, dim] into [dim]."""
    if x.ndim == 3:
        x = x.squeeze(0)
    if x.ndim != 2:
        raise ValueError(f"Expected 2D token embeddings, got shape {tuple(x.shape)}")
    if mode == "mean":
        return x.mean(dim=0)
    if mode == "max":
        return x.max(dim=0).values
    raise ValueError(f"Unknown pooling mode: {mode}")


def extract_video_embedding(
    model: Qwen2_5_VLForConditionalGeneration,
    processor: AutoProcessor,
    video_path: str,
    fps: float,
    min_frames: int,
    max_frames: int,
    frame_size: int,
    pooling: str,
) -> torch.Tensor:
    messages = make_message(video_path, fps, min_frames, max_frames, frame_size)
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    image_inputs, video_inputs, video_kwargs = process_vision_info(messages, return_video_kwargs=True)
    video_kwargs = fix_video_kwargs(video_kwargs)

    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
        **video_kwargs,
    ).to(model.device)

    if "pixel_values_videos" not in inputs or inputs["pixel_values_videos"] is None:
        raise ValueError(f"No video tensor produced for {video_path}")
    if "video_grid_thw" not in inputs or inputs["video_grid_thw"] is None:
        raise ValueError(f"No video_grid_thw produced for {video_path}")

    with torch.inference_mode():
        # This is the important part: visual-side video features before text decoding/generation.
        video_outputs = model.get_video_features(
            pixel_values_videos=inputs["pixel_values_videos"],
            video_grid_thw=inputs["video_grid_thw"],
        )

    # HF returns pooler_output as a tuple/list, one tensor per video in the batch.
    pooled = video_outputs.pooler_output
    if isinstance(pooled, (tuple, list)):
        token_embeddings = pooled[0]
    else:
        token_embeddings = pooled

    embedding = pool_tokens(token_embeddings.detach().float().cpu(), pooling)

    del inputs, video_outputs, pooled, token_embeddings
    torch.cuda.empty_cache()

    return embedding


def save_payload(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, tmp_path)
    tmp_path.replace(path)


def main() -> None:
    args = parse_args()

    video_max_pixels = args.video_max_pixels or int(args.frame_size * args.frame_size * args.max_frames)
    os.environ["VIDEO_MAX_PIXELS"] = str(video_max_pixels)

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

    print(f"Rows to process: {len(df)}")
    print(f"Model: {args.model_id}")
    print(f"Video sampling: fps={args.fps}, min_frames={args.min_frames}, max_frames={args.max_frames}, frame_size={args.frame_size}")
    print(f"VIDEO_MAX_PIXELS={os.environ['VIDEO_MAX_PIXELS']}")

    gc.collect()
    torch.cuda.empty_cache()

    model_kwargs: dict[str, Any] = {
        "torch_dtype": torch.float16,
        "device_map": "auto",
    }
    processor_kwargs: dict[str, Any] = {
        "min_pixels": 128 * 28 * 28,
        "max_pixels": args.frame_size * args.frame_size,
    }
    if args.trust_remote_code:
        model_kwargs["trust_remote_code"] = True
        processor_kwargs["trust_remote_code"] = True

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(args.model_id, **model_kwargs)
    processor = AutoProcessor.from_pretrained(args.model_id, **processor_kwargs)
    model.eval()

    embeddings: list[torch.Tensor] = []
    labels: list[int] = []
    sample_ids: list[str] = []
    metadata: list[dict[str, Any]] = []
    errors: list[dict[str, str]] = []

    for _, row in tqdm(df.iterrows(), total=len(df)):
        video_path = str(row["video_path"])
        sample_id = str(row["sample_id"])

        try:
            if not Path(video_path).exists():
                raise FileNotFoundError(video_path)
            embedding = extract_video_embedding(
                model=model,
                processor=processor,
                video_path=video_path,
                fps=args.fps,
                min_frames=args.min_frames,
                max_frames=args.max_frames,
                frame_size=args.frame_size,
                pooling=args.pooling,
            )
            embeddings.append(embedding)
            labels.append(int(row["emotion_id"]))
            sample_ids.append(sample_id)
            metadata.append(row.to_dict())
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            errors.append({"sample_id": sample_id, "video_path": video_path, "error": "CUDA OOM"})
        except Exception as exc:
            errors.append({"sample_id": sample_id, "video_path": video_path, "error": repr(exc)})

        if embeddings and (len(embeddings) % args.batch_save_every == 0):
            payload = {
                "embeddings": torch.stack(embeddings),
                "labels": torch.tensor(labels, dtype=torch.long),
                "label_names": LABELS,
                "sample_ids": sample_ids,
                "metadata": metadata,
                "errors": errors,
                "config": vars(args) | {"video_max_pixels": video_max_pixels},
            }
            save_payload(args.output_pt, payload)

    if not embeddings:
        raise RuntimeError(f"No embeddings were extracted. First errors: {errors[:5]}")

    payload = {
        "embeddings": torch.stack(embeddings),
        "labels": torch.tensor(labels, dtype=torch.long),
        "label_names": LABELS,
        "sample_ids": sample_ids,
        "metadata": metadata,
        "errors": errors,
        "config": vars(args) | {"video_max_pixels": video_max_pixels},
    }
    save_payload(args.output_pt, payload)

    print(f"Saved embeddings: {args.output_pt}")
    print(f"Embedding tensor: {tuple(payload['embeddings'].shape)}")
    print(f"Labels: {tuple(payload['labels'].shape)}")
    print(f"Errors: {len(errors)}")
    if errors:
        error_path = args.output_pt.with_suffix(".errors.json")
        error_path.write_text(json.dumps(errors, indent=2), encoding="utf-8")
        print(f"Saved errors: {error_path}")


if __name__ == "__main__":
    main()
