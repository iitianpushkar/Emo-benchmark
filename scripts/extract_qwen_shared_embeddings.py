#!/usr/bin/env python3
"""Extract shared Qwen2.5-VL video+utterance embeddings for MELD.

This script passes both the MELD video clip and its corresponding utterance
through Qwen2.5-VL, then pools hidden states from the multimodal transformer.
It does not call model.generate() and does not train Qwen.

Compared with extract_qwen_embeddings.py:
  - extract_qwen_embeddings.py saves video-only visual features from get_video_features(...)
  - this script saves shared/contextual video+text hidden-state embeddings
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
    parser = argparse.ArgumentParser(description="Extract shared Qwen2.5-VL video+utterance embeddings from MELD.")
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
        "--layer",
        type=int,
        default=-1,
        help="Hidden-state layer to pool. -1 means final transformer layer before LM head.",
    )
    parser.add_argument(
        "--pooling",
        choices=["last", "mean", "max"],
        default="last",
        help="How to pool sequence hidden states into one vector per sample. Default 'last' is task-conditioned.",
    )
    parser.add_argument(
        "--prompt-style",
        choices=["emotion_task", "utterance_only"],
        default="emotion_task",
        help="Text prompt used with each video. emotion_task includes the seven candidate emotions.",
    )
    parser.add_argument(
        "--no-generation-prompt",
        action="store_true",
        help="Do not append Qwen's assistant-generation marker before the forward pass.",
    )
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Pass trust_remote_code=True to from_pretrained if needed by a model variant.",
    )
    return parser.parse_args()


def make_prompt(utterance: str, prompt_style: str) -> str:
    if prompt_style == "utterance_only":
        return f"Utterance: {utterance}"
    if prompt_style == "emotion_task":
        return (
            "Task: infer the speaker's emotion from the video and utterance as exactly one of:\n"
            "Emotion choices: anger, disgust, fear, joy, neutral, sadness, surprise.\n"
            f"Utterance: {utterance}\n"
            "Focus on facial expression, body cues, scene context, and wording."
        )
    raise ValueError(f"Unknown prompt style: {prompt_style}")


def make_message(
    video_path: str,
    utterance: str,
    fps: float,
    min_frames: int,
    max_frames: int,
    frame_size: int,
    prompt_style: str,
) -> list[dict[str, Any]]:
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
                {"type": "text", "text": make_prompt(utterance, prompt_style)},
            ],
        }
    ]


def fix_video_kwargs(video_kwargs: dict[str, Any]) -> dict[str, Any]:
    fixed = dict(video_kwargs)
    # Compatibility fix for qwen-vl-utils / transformers combinations.
    if "fps" in fixed and isinstance(fixed["fps"], list) and len(fixed["fps"]) == 1:
        fixed["fps"] = fixed["fps"][0]
    return fixed


def pool_hidden_states(hidden: torch.Tensor, attention_mask: torch.Tensor, mode: str) -> torch.Tensor:
    """Pool [1, seq_len, hidden_dim] into [hidden_dim] using non-padding tokens."""
    if hidden.ndim != 3 or hidden.shape[0] != 1:
        raise ValueError(f"Expected hidden shape [1, seq_len, dim], got {tuple(hidden.shape)}")

    mask = attention_mask.bool().squeeze(0)
    token_hidden = hidden.squeeze(0)[mask]
    if token_hidden.numel() == 0:
        raise ValueError("No valid tokens found for pooling")

    if mode == "mean":
        return token_hidden.mean(dim=0)
    if mode == "max":
        return token_hidden.max(dim=0).values
    if mode == "last":
        return token_hidden[-1]
    raise ValueError(f"Unknown pooling mode: {mode}")


def forward_multimodal_hidden_states(
    model: Qwen2_5_VLForConditionalGeneration,
    inputs: dict[str, torch.Tensor],
) -> tuple[torch.Tensor, ...]:
    """Return shared transformer hidden states before Qwen's LM head when possible."""
    base_model = getattr(model, "model", None)
    if base_model is not None:
        outputs = base_model(
            **inputs,
            output_hidden_states=True,
            use_cache=False,
            return_dict=True,
        )
        return outputs.hidden_states

    # Fallback for unusual wrappers. logits_to_keep=1 reduces LM-head memory on newer transformers.
    try:
        outputs = model(
            **inputs,
            output_hidden_states=True,
            use_cache=False,
            return_dict=True,
            logits_to_keep=1,
        )
    except TypeError:
        outputs = model(
            **inputs,
            output_hidden_states=True,
            use_cache=False,
            return_dict=True,
        )
    return outputs.hidden_states


def extract_shared_embedding(
    model: Qwen2_5_VLForConditionalGeneration,
    processor: AutoProcessor,
    video_path: str,
    utterance: str,
    fps: float,
    min_frames: int,
    max_frames: int,
    frame_size: int,
    layer: int,
    pooling: str,
    prompt_style: str,
    add_generation_prompt: bool,
) -> torch.Tensor:
    messages = make_message(video_path, utterance, fps, min_frames, max_frames, frame_size, prompt_style)
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=add_generation_prompt)
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

    with torch.inference_mode():
        hidden_states = forward_multimodal_hidden_states(model, inputs)

    hidden = hidden_states[layer]
    embedding = pool_hidden_states(hidden.detach().float().cpu(), inputs["attention_mask"].detach().cpu(), pooling)

    del inputs, hidden_states, hidden
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

    required = ["sample_id", "video_path", "utterance", "emotion", "emotion_id"]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Index CSV is missing columns: {missing}")

    print(f"Rows to process: {len(df)}")
    print(f"Model: {args.model_id}")
    print("Embedding type: shared video+utterance hidden states")
    print(f"Layer: {args.layer}; pooling: {args.pooling}")
    print(f"Prompt style: {args.prompt_style}; add_generation_prompt={not args.no_generation_prompt}")
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
        utterance = str(row["utterance"])
        sample_id = str(row["sample_id"])

        try:
            if not Path(video_path).exists():
                raise FileNotFoundError(video_path)
            embedding = extract_shared_embedding(
                model=model,
                processor=processor,
                video_path=video_path,
                utterance=utterance,
                fps=args.fps,
                min_frames=args.min_frames,
                max_frames=args.max_frames,
                frame_size=args.frame_size,
                layer=args.layer,
                pooling=args.pooling,
                prompt_style=args.prompt_style,
                add_generation_prompt=not args.no_generation_prompt,
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
                "config": vars(args) | {"video_max_pixels": video_max_pixels, "embedding_type": "shared_video_text"},
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
        "config": vars(args) | {"video_max_pixels": video_max_pixels, "embedding_type": "shared_video_text"},
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
