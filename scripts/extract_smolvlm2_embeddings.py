#!/usr/bin/env python3
"""Extract frozen SmolVLM2 shared video-text embeddings for MELD.

This benchmark samples at 6 FPS by default (up to 64 frames), while the official
checkpoint default is 1 FPS. Each frame is processed at the checkpoint-native
384-pixel square resolution. Frames are decoded with Decord using the official
uniform sampling rule, while resize, normalization, padding, and token
construction remain owned by the checkpoint processor. The final shared
language hidden state is pooled before the LM head; generation is never called
and the SmolVLM2 weights remain frozen.
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
import transformers
from decord import VideoReader, cpu
from tqdm.auto import tqdm
from transformers import AutoModelForImageTextToText, AutoProcessor

LABELS = ["anger", "disgust", "fear", "joy", "neutral", "sadness", "surprise"]
OFFICIAL_MODEL_ID = "HuggingFaceTB/SmolVLM2-2.2B-Instruct"
OFFICIAL_FPS = 1.0
DEFAULT_BENCHMARK_FPS = 6.0
OFFICIAL_MAX_FRAMES = 64
OFFICIAL_FRAME_SIZE = 384
OFFICIAL_IMAGE_SEQ_LEN = 81
RESUME_CONFIG_KEYS = (
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract shared SmolVLM2 video-text embeddings from MELD.")
    parser.add_argument("--index-csv", type=Path, required=True, help="CSV produced by prepare_meld.py.")
    parser.add_argument("--output-pt", type=Path, required=True, help="Destination embedding .pt file.")
    parser.add_argument("--model-id", default=OFFICIAL_MODEL_ID, help="Hugging Face model id.")
    parser.add_argument(
        "--fps",
        type=float,
        default=DEFAULT_BENCHMARK_FPS,
        help="Target sampling FPS. Benchmark default: 6; official checkpoint default: 1.",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=OFFICIAL_MAX_FRAMES,
        help="Maximum sampled frames. The official checkpoint default is 64.",
    )
    parser.add_argument(
        "--skip-secs",
        type=float,
        default=1.0,
        help="Official sampler's optional boundary skip for sufficiently long videos.",
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
    parser.add_argument(
        "--compute-dtype",
        choices=["auto", "float16", "bfloat16"],
        default="auto",
        help="Model compute dtype. auto uses BF16 when supported, otherwise FP16.",
    )
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
        content.append({"type": "video"})
    content.append({"type": "text", "text": make_prompt(utterance, style, mode)})
    return [{"role": "user", "content": content}]


def official_sample_indices(
    total_frames: int,
    source_fps: float,
    target_fps: float,
    max_frames: int,
    skip_secs: float,
) -> np.ndarray:
    """Reproduce SmolVLMVideoProcessor.sample_frames for Decord input."""
    if total_frames <= 0 or not math.isfinite(source_fps) or source_fps <= 0:
        raise ValueError(f"Invalid video metadata: frames={total_frames}, fps={source_fps}")

    duration = total_frames / source_fps
    estimated_frames = int(round(target_fps * duration))
    desired_frames = max(1, min(estimated_frames, max_frames))

    start_idx = 0
    end_idx = total_frames - 1
    if skip_secs > 0 and (duration - 2 * skip_secs) > (max_frames * target_fps):
        start_idx = max(0, int(skip_secs * source_fps))
        end_idx = min(total_frames - 1, int(total_frames - skip_secs * source_fps))
        if start_idx >= end_idx:
            start_idx, end_idx = 0, total_frames - 1

    return np.unique(np.linspace(start_idx, end_idx, desired_frames, dtype=np.int64))


def sample_video(
    video_path: str,
    target_fps: float,
    max_frames: int,
    skip_secs: float,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Decode only officially selected frames as uint8 [T, H, W, C]."""
    reader = VideoReader(video_path, ctx=cpu(0))
    total_frames = len(reader)
    source_fps = float(reader.get_avg_fps())
    indices = official_sample_indices(total_frames, source_fps, target_fps, max_frames, skip_secs)
    frames = reader.get_batch(indices).asnumpy()
    metadata = {
        "source_fps": source_fps,
        "source_frames": total_frames,
        "source_duration_seconds": total_frames / source_fps,
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
    """Pool on the accelerator and copy one independent float32 vector to CPU."""
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


def processor_setting(value: Any, key: str) -> int | None:
    if isinstance(value, dict):
        result = value.get(key)
    else:
        result = getattr(value, key, None)
    return int(result) if result is not None else None


def validate_official_processor(processor: Any, model_id: str) -> dict[str, int]:
    video_processor = getattr(processor, "video_processor", None)
    if video_processor is None:
        raise ValueError("Loaded processor has no SmolVLM video_processor")

    frame_size = processor_setting(getattr(video_processor, "max_image_size", None), "longest_edge")
    if frame_size is None:
        frame_size = processor_setting(getattr(video_processor, "size", None), "longest_edge")
    image_seq_len = int(getattr(processor, "image_seq_len"))

    if model_id == OFFICIAL_MODEL_ID:
        if frame_size != OFFICIAL_FRAME_SIZE or image_seq_len != OFFICIAL_IMAGE_SEQ_LEN:
            raise ValueError(
                "Official SmolVLM2 processor geometry changed unexpectedly: "
                f"frame_size={frame_size}, image_seq_len={image_seq_len}"
            )
    return {"frame_size": frame_size, "image_seq_len": image_seq_len}


def extract_embedding(
    model: torch.nn.Module,
    processor: Any,
    video_path: str | None,
    utterance: str | None,
    fps: float,
    max_frames: int,
    skip_secs: float,
    pooling: str,
    prompt_style: str,
    add_generation_prompt: bool,
    modality_mode: str,
    dtype: torch.dtype,
    expected_frame_size: int,
    image_seq_len: int,
) -> tuple[torch.Tensor, dict[str, Any]]:
    messages = make_message(utterance, prompt_style, modality_mode)
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=add_generation_prompt)
    decode_metadata: dict[str, Any] = {}
    frames = None
    if modality_mode in {"video_text", "video_only"}:
        if video_path is None:
            raise ValueError("video_path is required for video modes")
        frames, decode_metadata = sample_video(video_path, fps, max_frames, skip_secs)

    processor_kwargs: dict[str, Any] = {
        "text": [text],
        "padding": True,
        "return_tensors": "pt",
    }
    if frames is not None:
        processor_kwargs["videos"] = [frames]
        processor_kwargs["do_sample_frames"] = False

    inputs = processor(**processor_kwargs)
    if frames is not None:
        pixel_values = inputs.get("pixel_values")
        expected_shape = (expected_frame_size, expected_frame_size)
        if pixel_values is None or tuple(pixel_values.shape[-2:]) != expected_shape:
            shape = None if pixel_values is None else tuple(pixel_values.shape)
            raise ValueError(f"Unexpected processed video shape: expected {expected_shape}, got {shape}")
        image_token_id = int(model.config.image_token_id)
        actual_image_tokens = int((inputs["input_ids"] == image_token_id).sum().item())
        expected_image_tokens = len(frames) * image_seq_len
        if actual_image_tokens != expected_image_tokens:
            raise ValueError(
                "Video feature/token mismatch: "
                f"expected {expected_image_tokens} image tokens, got {actual_image_tokens}"
            )
        decode_metadata |= {
            "processed_frame_height": int(pixel_values.shape[-2]),
            "processed_frame_width": int(pixel_values.shape[-1]),
            "visual_tokens": actual_image_tokens,
            "visual_tokens_per_frame": image_seq_len,
        }
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


def load_resume_payload(
    path: Path,
    dtype: torch.dtype,
    retry_errors: bool,
    expected_hidden_size: int,
    args: argparse.Namespace,
    resolved_compute_dtype: torch.dtype,
):
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
            f"does not match SmolVLM2 hidden size {expected_hidden_size}"
        )
    stored_config = payload.get("config", {})
    for key in RESUME_CONFIG_KEYS:
        if key not in stored_config:
            raise ValueError(f"Cannot safely resume {path}; stored config is missing {key}")
        if stored_config[key] != getattr(args, key):
            raise ValueError(
                f"Cannot resume {path}; {key} changed from {stored_config[key]!r} to {getattr(args, key)!r}"
            )
    expected_runtime = {
        "resolved_compute_dtype": str(resolved_compute_dtype),
        "transformers_version": transformers.__version__,
    }
    for key, value in expected_runtime.items():
        if stored_config.get(key) != value:
            raise ValueError(
                f"Cannot resume {path}; {key} changed from {stored_config.get(key)!r} to {value!r}"
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
    processor_config: dict[str, int],
    resolved_compute_dtype: torch.dtype,
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
            "embedding_type": f"smolvlm2_shared_lm_{args.modality_mode}",
            "hidden_size": hidden_size,
            "native_frame_processing": True,
            "processor_frame_size": processor_config["frame_size"],
            "image_seq_len": processor_config["image_seq_len"],
            "resolved_compute_dtype": str(resolved_compute_dtype),
            "transformers_version": transformers.__version__,
            "official_sampling_defaults_used": (
                args.fps == OFFICIAL_FPS and args.max_frames == OFFICIAL_MAX_FRAMES
            ),
        },
    }


def main() -> None:
    args = parse_args()
    if args.layer != -1:
        raise ValueError("SmolVLM2 extractor supports only --layer -1 to avoid retaining every hidden layer")
    if args.fps <= 0 or args.max_frames <= 0 or args.skip_secs < 0:
        raise ValueError("Require fps > 0, max_frames > 0, and skip_secs >= 0")

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
    print(f"Sampling: fps={args.fps}, max_frames={args.max_frames}, skip_secs={args.skip_secs}")

    model = AutoModelForImageTextToText.from_pretrained(
        args.model_id,
        torch_dtype=dtype,
        device_map="auto",
        low_cpu_mem_usage=True,
    )
    processor = AutoProcessor.from_pretrained(args.model_id)
    processor_config = validate_official_processor(processor, args.model_id)
    print(
        "Frame preprocessing: checkpoint-native "
        f"{processor_config['frame_size']}x{processor_config['frame_size']}; "
        f"visual tokens/frame={processor_config['image_seq_len']}"
    )
    if args.fps != OFFICIAL_FPS or args.max_frames != OFFICIAL_MAX_FRAMES:
        print("WARNING: temporal sampling differs from the official checkpoint defaults (1 FPS, max 64 frames).")

    model.eval()
    hidden_size = int(model.config.text_config.hidden_size)
    print(f"Shared hidden size: {hidden_size}")

    output_dtype = saved_dtype(args.save_dtype)
    if args.resume:
        embeddings, labels, sample_ids, metadata, errors, done_ids = load_resume_payload(
            args.output_pt, output_dtype, args.retry_errors, hidden_size, args, dtype
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
                max_frames=args.max_frames,
                skip_secs=args.skip_secs,
                pooling=args.pooling,
                prompt_style=args.prompt_style,
                add_generation_prompt=not args.no_generation_prompt,
                modality_mode=args.modality_mode,
                dtype=dtype,
                expected_frame_size=processor_config["frame_size"],
                image_seq_len=processor_config["image_seq_len"],
            )
            if embedding.numel() != hidden_size or not torch.isfinite(embedding).all():
                raise ValueError(f"Invalid embedding shape/values: {tuple(embedding.shape)}")
            embeddings.append(embedding.to(dtype=output_dtype))
            labels.append(int(row["emotion_id"]))
            sample_ids.append(sample_id)
            metadata.append(row.to_dict() | decode_metadata | {"modality_mode": args.modality_mode})
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
                build_payload(
                    embeddings,
                    labels,
                    sample_ids,
                    metadata,
                    errors,
                    args,
                    hidden_size,
                    processor_config,
                    dtype,
                ),
            )

    if not embeddings:
        raise RuntimeError(f"No embeddings extracted. First errors: {errors[:5]}")
    payload = build_payload(
        embeddings, labels, sample_ids, metadata, errors, args, hidden_size, processor_config, dtype
    )
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
