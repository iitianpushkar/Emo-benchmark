#!/usr/bin/env python3
"""Extract frozen InternVL3 shared video-text embeddings for MELD.

Frames are sampled with Decord, processed with InternVL's native video
processor, and combined with the MELD utterance by the language transformer.
The script pools the final hidden state before the LM head; it never calls
model.generate() and never trains InternVL.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from decord import VideoReader, cpu
from tqdm.auto import tqdm
from transformers import AutoModelForImageTextToText, AutoProcessor

LABELS = ["anger", "disgust", "fear", "joy", "neutral", "sadness", "surprise"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract shared InternVL3 video-text embeddings from MELD.")
    parser.add_argument("--index-csv", type=Path, required=True, help="CSV produced by prepare_meld.py.")
    parser.add_argument("--output-pt", type=Path, required=True, help="Destination embedding .pt file.")
    parser.add_argument("--model-id", default="OpenGVLab/InternVL3-2B-hf", help="Hugging Face model id.")
    parser.add_argument("--fps", type=float, default=6.0, help="Target video sampling rate.")
    parser.add_argument("--max-frames", type=int, default=64, help="Maximum frames per video.")
    parser.add_argument("--min-frames", type=int, default=4, help="Minimum frames for sufficiently long videos.")
    parser.add_argument(
        "--frame-size",
        type=int,
        default=384,
        help="Compatibility argument. InternVL uses its checkpoint-native frame size (normally 384).",
    )
    parser.add_argument("--limit", type=int, default=None, help="Optional number of selected rows.")
    parser.add_argument("--start", type=int, default=0, help="Start row offset in the filtered index.")
    parser.add_argument("--batch-save-every", type=int, default=25, help="Checkpoint interval in samples.")
    parser.add_argument("--resume", action="store_true", help="Resume an existing output and skip recorded samples.")
    parser.add_argument("--retry-errors", action="store_true", help="Retry errors already recorded in the output.")
    parser.add_argument("--gc-every", type=int, default=10, help="Run Python/CUDA cleanup every N attempts.")
    parser.add_argument("--save-dtype", choices=["float16", "float32"], default="float32")
    parser.add_argument("--layer", type=int, default=-1, help="Only -1 is supported without retaining every layer.")
    parser.add_argument("--pooling", choices=["last", "mean", "max"], default="last")
    parser.add_argument("--prompt-style", choices=["emotion_task", "utterance_only"], default="emotion_task")
    parser.add_argument("--no-generation-prompt", action="store_true")
    parser.add_argument(
        "--modality-mode",
        choices=["video_text", "video_only", "text_only"],
        default="video_text",
    )
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument(
        "--compute-dtype",
        choices=["auto", "float16", "bfloat16"],
        default="auto",
        help="Model compute dtype. auto uses BF16 when supported, otherwise FP16.",
    )
    # Accepted for compatibility with run_qwen_shared_chunks.py.
    parser.add_argument("--video-max-pixels", type=int, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--skip-decord-precheck", action="store_true", help=argparse.SUPPRESS)
    return parser.parse_args()


def make_prompt(utterance: str | None, style: str, mode: str) -> str:
    if mode == "video_only":
        if style == "utterance_only":
            return "Infer the speaker's emotion from the video."
        return (
            "Task: infer the speaker's emotion from the video as exactly one of:\n"
            "Emotion choices: anger, disgust, fear, joy, neutral, sadness, surprise.\n"
            "Focus on facial expression, body cues, and scene context."
        )

    utterance = "" if utterance is None else utterance
    if style == "utterance_only":
        return f"Utterance: {utterance}"
    if mode == "text_only":
        return (
            "Task: infer the speaker's emotion from the utterance as exactly one of:\n"
            "Emotion choices: anger, disgust, fear, joy, neutral, sadness, surprise.\n"
            f"Utterance: {utterance}\n"
            "Focus on wording and conversational meaning."
        )
    return (
        "Task: infer the speaker's emotion from the video and utterance as exactly one of:\n"
        "Emotion choices: anger, disgust, fear, joy, neutral, sadness, surprise.\n"
        f"Utterance: {utterance}\n"
        "Focus on facial expression, body cues, scene context, and wording."
    )


def make_message(utterance: str | None, style: str, mode: str) -> list[dict[str, Any]]:
    content: list[dict[str, Any]] = []
    if mode in {"video_text", "video_only"}:
        # The actual sampled frame array is supplied separately to processor(...).
        content.append({"type": "video"})
    content.append({"type": "text", "text": make_prompt(utterance, style, mode)})
    return [{"role": "user", "content": content}]


def sample_video(video_path: str, target_fps: float, min_frames: int, max_frames: int) -> tuple[np.ndarray, dict[str, Any]]:
    """Decode only selected frames and return uint8 [T, H, W, C]."""
    reader = VideoReader(video_path, ctx=cpu(0))
    total_frames = len(reader)
    source_fps = float(reader.get_avg_fps())
    if total_frames <= 0 or not math.isfinite(source_fps) or source_fps <= 0:
        raise ValueError(f"Invalid video metadata: frames={total_frames}, fps={source_fps}")

    step = source_fps / target_fps
    indices = np.arange(0, total_frames, step).round().astype(np.int64)
    indices = np.unique(np.clip(indices, 0, total_frames - 1))

    desired_min = min(min_frames, total_frames)
    if len(indices) < desired_min:
        indices = np.linspace(0, total_frames - 1, desired_min).round().astype(np.int64)
    if len(indices) > max_frames:
        keep = np.linspace(0, len(indices) - 1, max_frames).round().astype(np.int64)
        indices = indices[keep]
    indices = np.unique(indices)

    frames = reader.get_batch(indices).asnumpy()
    metadata = {
        "source_fps": source_fps,
        "source_frames": total_frames,
        "sampled_frames": int(len(indices)),
        "sampled_indices": indices.tolist(),
    }
    del reader
    return frames, metadata


def compute_dtype(name: str) -> torch.dtype:
    if name == "bfloat16":
        return torch.bfloat16
    if name == "float16":
        return torch.float16
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return torch.float16


def saved_dtype(name: str) -> torch.dtype:
    return torch.float16 if name == "float16" else torch.float32


def cleanup_memory() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def input_device(model: torch.nn.Module) -> torch.device:
    return model.get_input_embeddings().weight.device


def move_inputs(inputs: dict[str, Any], device: torch.device, dtype: torch.dtype) -> dict[str, Any]:
    moved = {}
    for key, value in inputs.items():
        if not torch.is_tensor(value):
            moved[key] = value
        elif value.is_floating_point():
            moved[key] = value.to(device=device, dtype=dtype)
        else:
            moved[key] = value.to(device=device)
    return moved


def pool_hidden(hidden: torch.Tensor, attention_mask: torch.Tensor, mode: str) -> torch.Tensor:
    """Pool on the accelerator, then copy only one independent vector to CPU."""
    mask = attention_mask[0].bool()
    token_hidden = hidden[0]
    if mode == "last":
        pooled = token_hidden[mask.nonzero(as_tuple=False)[-1, 0]]
    elif mode == "mean":
        pooled = token_hidden[mask].mean(dim=0)
    elif mode == "max":
        pooled = token_hidden[mask].max(dim=0).values
    else:
        raise ValueError(f"Unknown pooling mode: {mode}")
    return pooled.detach().to(device="cpu", dtype=torch.float32).clone()


def extract_embedding(
    model: torch.nn.Module,
    processor: Any,
    video_path: str | None,
    utterance: str | None,
    fps: float,
    min_frames: int,
    max_frames: int,
    pooling: str,
    prompt_style: str,
    add_generation_prompt: bool,
    modality_mode: str,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, dict[str, Any]]:
    messages = make_message(utterance, prompt_style, modality_mode)
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=add_generation_prompt)
    decode_metadata: dict[str, Any] = {}
    frames = None
    if modality_mode in {"video_text", "video_only"}:
        if video_path is None:
            raise ValueError("video_path is required for video modes")
        frames, decode_metadata = sample_video(video_path, fps, min_frames, max_frames)

    processor_kwargs: dict[str, Any] = {
        "text": [text],
        "padding": True,
        "return_tensors": "pt",
    }
    if frames is not None:
        processor_kwargs["videos"] = [frames]
        processor_kwargs["do_sample_frames"] = False

    inputs = processor(**processor_kwargs)
    inputs = move_inputs(inputs, input_device(model), dtype)
    try:
        with torch.inference_mode():
            outputs = model.model(**inputs, use_cache=False, return_dict=True)
            embedding = pool_hidden(outputs.last_hidden_state, inputs["attention_mask"], pooling)
    finally:
        del messages, text, processor_kwargs, inputs
        if frames is not None:
            del frames
        if "outputs" in locals():
            del outputs
    return embedding, decode_metadata


def save_payload(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def load_resume_payload(path: Path, dtype: torch.dtype, retry_errors: bool, expected_hidden_size: int):
    if not path.exists():
        return [], [], [], [], [], set()
    payload = torch.load(path, map_location="cpu", weights_only=False)
    for key in ("embeddings", "labels", "sample_ids"):
        if key not in payload:
            raise ValueError(f"Cannot resume from {path}; missing {key}")
    embedding_tensor = payload["embeddings"].cpu().to(dtype=dtype)
    label_tensor = payload["labels"].cpu().long()
    sample_ids = [str(value) for value in payload["sample_ids"]]
    if embedding_tensor.shape[0] != len(sample_ids) or label_tensor.shape[0] != len(sample_ids):
        raise ValueError(f"Cannot resume from {path}; payload lengths differ")
    if embedding_tensor.ndim != 2 or embedding_tensor.shape[1] != expected_hidden_size:
        raise ValueError(
            f"Cannot resume from {path}; embedding shape {tuple(embedding_tensor.shape)} "
            f"does not match InternVL hidden size {expected_hidden_size}"
        )
    metadata = list(payload.get("metadata", [{} for _ in sample_ids]))
    errors = list(payload.get("errors", []))
    done_ids = set(sample_ids)
    if not retry_errors:
        done_ids.update(str(error["sample_id"]) for error in errors if error.get("sample_id"))
    embeddings = [embedding_tensor[i] for i in range(embedding_tensor.shape[0])]
    return embeddings, label_tensor.tolist(), sample_ids, metadata, errors, done_ids


def build_payload(
    embeddings: list[torch.Tensor],
    labels: list[int],
    sample_ids: list[str],
    metadata: list[dict[str, Any]],
    errors: list[dict[str, str]],
    args: argparse.Namespace,
    hidden_size: int,
) -> dict[str, Any]:
    return {
        "embeddings": torch.stack(embeddings),
        "labels": torch.tensor(labels, dtype=torch.long),
        "label_names": LABELS,
        "sample_ids": sample_ids,
        "metadata": metadata,
        "errors": errors,
        "config": vars(args)
        | {
            "embedding_type": f"internvl3_shared_lm_{args.modality_mode}",
            "hidden_size": hidden_size,
            "native_frame_processing": True,
        },
    }


def main() -> None:
    args = parse_args()
    if args.layer != -1:
        raise ValueError("InternVL3 extractor currently supports only --layer -1 to avoid storing every hidden layer")
    if args.fps <= 0 or args.min_frames <= 0 or args.max_frames < args.min_frames:
        raise ValueError("Require fps > 0 and 0 < min_frames <= max_frames")

    df = pd.read_csv(args.index_csv)
    include_video = args.modality_mode in {"video_text", "video_only"}
    include_utterance = args.modality_mode in {"video_text", "text_only"}
    if include_video and "video_exists" in df.columns:
        df = df[df["video_exists"].astype(bool)].copy()
    if args.start:
        df = df.iloc[args.start:].copy()
    if args.limit is not None:
        df = df.head(args.limit).copy()
    df = df.reset_index(drop=True)

    required = ["sample_id", "video_path", "utterance", "emotion_id"]
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise ValueError(f"Index CSV is missing columns: {missing}")

    dtype = compute_dtype(args.compute_dtype)
    print(f"Rows to process: {len(df)}")
    print(f"Model: {args.model_id}")
    print(f"Embedding: final shared video-text hidden state; mode={args.modality_mode}")
    print(f"Pooling: {args.pooling}; compute dtype: {dtype}; save dtype: {args.save_dtype}")
    print(f"Sampling: fps={args.fps}, min_frames={args.min_frames}, max_frames={args.max_frames}")
    print("Frame preprocessing: InternVL checkpoint-native video processor")

    model = AutoModelForImageTextToText.from_pretrained(
        args.model_id,
        torch_dtype=dtype,
        device_map="auto",
        low_cpu_mem_usage=True,
        trust_remote_code=args.trust_remote_code,
    )
    processor = AutoProcessor.from_pretrained(args.model_id, trust_remote_code=args.trust_remote_code)
    model.eval()
    hidden_size = int(model.config.text_config.hidden_size)
    print(f"Shared hidden size: {hidden_size}")

    output_dtype = saved_dtype(args.save_dtype)
    if args.resume:
        embeddings, labels, sample_ids, metadata, errors, done_ids = load_resume_payload(
            args.output_pt, output_dtype, args.retry_errors, hidden_size
        )
        print(f"Resume enabled: loaded {len(sample_ids)} embeddings and {len(errors)} errors")
    else:
        embeddings, labels, sample_ids, metadata, errors, done_ids = [], [], [], [], [], set()

    attempts = 0
    for _, row in tqdm(df.iterrows(), total=len(df)):
        sample_id = str(row["sample_id"])
        if sample_id in done_ids:
            continue
        video_path = str(row["video_path"])
        try:
            if include_video and not Path(video_path).exists():
                raise FileNotFoundError(video_path)
            embedding, decode_metadata = extract_embedding(
                model=model,
                processor=processor,
                video_path=video_path if include_video else None,
                utterance=str(row["utterance"]) if include_utterance else None,
                fps=args.fps,
                min_frames=args.min_frames,
                max_frames=args.max_frames,
                pooling=args.pooling,
                prompt_style=args.prompt_style,
                add_generation_prompt=not args.no_generation_prompt,
                modality_mode=args.modality_mode,
                dtype=dtype,
            )
            if embedding.numel() != hidden_size or not torch.isfinite(embedding).all():
                raise ValueError(f"Invalid embedding shape/values: {tuple(embedding.shape)}")
            embeddings.append(embedding.to(dtype=output_dtype))
            labels.append(int(row["emotion_id"]))
            sample_ids.append(sample_id)
            row_metadata = row.to_dict() | decode_metadata | {"modality_mode": args.modality_mode}
            metadata.append(row_metadata)
            done_ids.add(sample_id)
        except torch.cuda.OutOfMemoryError:
            errors.append({"sample_id": sample_id, "video_path": video_path, "error": "CUDA OOM"})
            if not args.retry_errors:
                done_ids.add(sample_id)
            cleanup_memory()
        except Exception as exc:
            errors.append({"sample_id": sample_id, "video_path": video_path, "error": repr(exc)})
            if not args.retry_errors:
                done_ids.add(sample_id)

        attempts += 1
        if args.gc_every > 0 and attempts % args.gc_every == 0:
            cleanup_memory()
        if embeddings and len(embeddings) % args.batch_save_every == 0:
            save_payload(
                args.output_pt,
                build_payload(embeddings, labels, sample_ids, metadata, errors, args, hidden_size),
            )

    if not embeddings:
        raise RuntimeError(f"No embeddings extracted. First errors: {errors[:5]}")
    payload = build_payload(embeddings, labels, sample_ids, metadata, errors, args, hidden_size)
    save_payload(args.output_pt, payload)
    print(f"Saved embeddings: {args.output_pt}")
    print(f"Embedding tensor: {tuple(payload['embeddings'].shape)}")
    print(f"Errors: {len(errors)}")
    if errors:
        error_path = args.output_pt.with_suffix(".errors.json")
        error_path.write_text(json.dumps(errors, indent=2), encoding="utf-8")
        print(f"Saved errors: {error_path}")


if __name__ == "__main__":
    main()
