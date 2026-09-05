#!/usr/bin/env python3
"""Extract shared Qwen2.5-VL video+utterance embeddings for MELD.

This script passes MELD inputs through Qwen2.5-VL, then pools hidden states
from the multimodal transformer.
It does not call model.generate() and does not train Qwen.

Compared with extract_qwen_embeddings.py:
  - extract_qwen_embeddings.py saves video-only visual features from get_video_features(...)
  - this script saves shared/contextual video+text hidden-state embeddings
  - this script can also extract video-only or text-only LM-space embeddings
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
    parser.add_argument("--resume", action="store_true", help="Resume from an existing output .pt file and skip completed samples.")
    parser.add_argument(
        "--retry-errors",
        action="store_true",
        help="When resuming, retry samples that were saved as errors instead of skipping them.",
    )
    parser.add_argument("--gc-every", type=int, default=10, help="Run Python/CUDA cleanup after this many samples.")
    parser.add_argument(
        "--save-dtype",
        choices=["float16", "float32"],
        default="float16",
        help="Dtype used for saved embeddings. train_mlp.py converts them back to float32 while training.",
    )
    parser.add_argument(
        "--skip-decord-precheck",
        action="store_true",
        help="Do not pre-open videos with decord before Qwen processing. Precheck skips corrupted MP4s earlier.",
    )
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
        "--modality-mode",
        choices=["video_text", "video_only", "text_only"],
        default="video_text",
        help=(
            "Input condition for LM-space extraction. video_text uses the matched clip and utterance; "
            "video_only removes the utterance; text_only removes the video."
        ),
    )
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Pass trust_remote_code=True to from_pretrained if needed by a model variant.",
    )
    return parser.parse_args()


def make_prompt(utterance: str | None, prompt_style: str, modality_mode: str) -> str:
    if modality_mode == "video_only":
        if prompt_style == "utterance_only":
            return "Infer the speaker's emotion from the video."
        if prompt_style == "emotion_task":
            return (
                "Task: infer the speaker's emotion from the video as exactly one of:\n"
                "Emotion choices: anger, disgust, fear, joy, neutral, sadness, surprise.\n"
                "Focus on facial expression, body cues, and scene context."
            )
        raise ValueError(f"Unknown prompt style: {prompt_style}")

    utterance = "" if utterance is None else utterance
    if modality_mode == "text_only":
        if prompt_style == "utterance_only":
            return f"Utterance: {utterance}"
        if prompt_style == "emotion_task":
            return (
                "Task: infer the speaker's emotion from the utterance as exactly one of:\n"
                "Emotion choices: anger, disgust, fear, joy, neutral, sadness, surprise.\n"
                f"Utterance: {utterance}\n"
                "Focus on wording and conversational meaning."
            )
        raise ValueError(f"Unknown prompt style: {prompt_style}")

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
    video_path: str | None,
    utterance: str | None,
    fps: float,
    min_frames: int,
    max_frames: int,
    frame_size: int,
    prompt_style: str,
    modality_mode: str,
) -> list[dict[str, Any]]:
    content: list[dict[str, Any]] = []
    if modality_mode in {"video_text", "video_only"}:
        if video_path is None:
            raise ValueError("video_path is required for video_text and video_only modes")
        content.append(
            {
                "type": "video",
                "video": video_path,
                "fps": fps,
                "min_frames": min_frames,
                "max_frames": max_frames,
                "resized_height": frame_size,
                "resized_width": frame_size,
            }
        )
    content.append({"type": "text", "text": make_prompt(utterance, prompt_style, modality_mode)})
    return [
        {
            "role": "user",
            "content": content,
        }
    ]


def fix_video_kwargs(video_kwargs: dict[str, Any]) -> dict[str, Any]:
    fixed = dict(video_kwargs)
    # Compatibility fix for qwen-vl-utils / transformers combinations.
    if "fps" in fixed and isinstance(fixed["fps"], list) and len(fixed["fps"]) == 1:
        fixed["fps"] = fixed["fps"][0]
    return fixed


def precheck_video_with_decord(video_path: str) -> None:
    """Fail early on corrupt MP4s so qwen-vl-utils does not fall back to slower torchvision decoding."""
    try:
        from decord import VideoReader, cpu
    except ImportError:
        return

    vr = VideoReader(video_path, ctx=cpu(0))
    if len(vr) <= 0:
        raise ValueError(f"No decodable frames found: {video_path}")
    del vr


def cleanup_memory() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def save_embedding_dtype(name: str) -> torch.dtype:
    if name == "float16":
        return torch.float16
    if name == "float32":
        return torch.float32
    raise ValueError(f"Unknown save dtype: {name}")


def load_resume_payload(
    path: Path,
    output_dtype: torch.dtype,
    retry_errors: bool,
) -> tuple[list[torch.Tensor], list[int], list[str], list[dict[str, Any]], list[dict[str, str]], set[str]]:
    if not path.exists():
        return [], [], [], [], [], set()

    payload = torch.load(path, map_location="cpu", weights_only=False)
    if "embeddings" not in payload or "labels" not in payload or "sample_ids" not in payload:
        raise ValueError(f"Cannot resume from {path}; missing embeddings, labels, or sample_ids")

    embeddings_tensor = payload["embeddings"].cpu().to(dtype=output_dtype)
    labels_tensor = payload["labels"].cpu().long()
    sample_ids = [str(item) for item in payload.get("sample_ids", [])]
    metadata = list(payload.get("metadata", [{} for _ in sample_ids]))
    errors = list(payload.get("errors", []))

    if embeddings_tensor.shape[0] != len(sample_ids):
        raise ValueError(
            f"Cannot resume from {path}; embeddings rows ({embeddings_tensor.shape[0]}) "
            f"do not match sample_ids ({len(sample_ids)})"
        )
    if labels_tensor.shape[0] != len(sample_ids):
        raise ValueError(
            f"Cannot resume from {path}; label rows ({labels_tensor.shape[0]}) "
            f"do not match sample_ids ({len(sample_ids)})"
        )

    done_ids = set(sample_ids)
    if not retry_errors:
        done_ids.update(str(error.get("sample_id")) for error in errors if error.get("sample_id"))

    embeddings = [embeddings_tensor[i] for i in range(embeddings_tensor.shape[0])]
    labels = labels_tensor.tolist()
    return embeddings, labels, sample_ids, metadata, errors, done_ids


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


def forward_multimodal_hidden_state(
    model: Qwen2_5_VLForConditionalGeneration,
    inputs: dict[str, torch.Tensor],
    layer: int,
) -> torch.Tensor:
    """Return one shared transformer hidden-state tensor before Qwen's LM head."""
    base_model = getattr(model, "model", None)
    if base_model is not None:
        outputs = base_model(
            **inputs,
            output_hidden_states=(layer != -1),
            use_cache=False,
            return_dict=True,
        )
        if layer == -1:
            return outputs.last_hidden_state
        return outputs.hidden_states[layer]

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
    return outputs.hidden_states[layer]


def extract_shared_embedding(
    model: Qwen2_5_VLForConditionalGeneration,
    processor: AutoProcessor,
    video_path: str | None,
    utterance: str | None,
    fps: float,
    min_frames: int,
    max_frames: int,
    frame_size: int,
    layer: int,
    pooling: str,
    prompt_style: str,
    add_generation_prompt: bool,
    modality_mode: str,
) -> torch.Tensor:
    messages = make_message(video_path, utterance, fps, min_frames, max_frames, frame_size, prompt_style, modality_mode)
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

    try:
        with torch.inference_mode():
            hidden = forward_multimodal_hidden_state(model, inputs, layer)

        embedding = pool_hidden_states(hidden.detach().float().cpu(), inputs["attention_mask"].detach().cpu(), pooling)
    finally:
        del messages, text, image_inputs, video_inputs, video_kwargs, inputs
        if "hidden" in locals():
            del hidden
        cleanup_memory()

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
    include_video = args.modality_mode in {"video_text", "video_only"}
    include_utterance = args.modality_mode in {"video_text", "text_only"}

    if include_video and "video_exists" in df.columns:
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
    print(f"Embedding type: shared LM hidden states; modality_mode={args.modality_mode}")
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

    output_dtype = save_embedding_dtype(args.save_dtype)

    if args.resume:
        embeddings, labels, sample_ids, metadata, errors, done_ids = load_resume_payload(
            args.output_pt, output_dtype=output_dtype, retry_errors=args.retry_errors
        )
        print(f"Resume enabled: loaded {len(sample_ids)} embeddings and {len(errors)} previous errors")
        print(f"Resume skip set: {len(done_ids)} sample ids")
    else:
        embeddings: list[torch.Tensor] = []
        labels: list[int] = []
        sample_ids: list[str] = []
        metadata: list[dict[str, Any]] = []
        errors: list[dict[str, str]] = []
        done_ids: set[str] = set()

    for _, row in tqdm(df.iterrows(), total=len(df)):
        video_path = str(row["video_path"])
        utterance = str(row["utterance"])
        sample_id = str(row["sample_id"])

        if sample_id in done_ids:
            continue

        try:
            if include_video and not Path(video_path).exists():
                raise FileNotFoundError(video_path)
            if include_video and not args.skip_decord_precheck:
                precheck_video_with_decord(video_path)
            embedding = extract_shared_embedding(
                model=model,
                processor=processor,
                video_path=video_path if include_video else None,
                utterance=utterance if include_utterance else None,
                fps=args.fps,
                min_frames=args.min_frames,
                max_frames=args.max_frames,
                frame_size=args.frame_size,
                layer=args.layer,
                pooling=args.pooling,
                prompt_style=args.prompt_style,
                add_generation_prompt=not args.no_generation_prompt,
                modality_mode=args.modality_mode,
            )
            embeddings.append(embedding.to(dtype=output_dtype))
            labels.append(int(row["emotion_id"]))
            sample_ids.append(sample_id)
            row_metadata = row.to_dict()
            row_metadata["modality_mode"] = args.modality_mode
            row_metadata["include_video"] = include_video
            row_metadata["include_utterance"] = include_utterance
            metadata.append(row_metadata)
            done_ids.add(sample_id)
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            errors.append({"sample_id": sample_id, "video_path": video_path, "error": "CUDA OOM"})
            if not args.retry_errors:
                done_ids.add(sample_id)
        except Exception as exc:
            errors.append({"sample_id": sample_id, "video_path": video_path, "error": repr(exc)})
            if not args.retry_errors:
                done_ids.add(sample_id)

        if args.gc_every > 0 and ((len(embeddings) + len(errors)) % args.gc_every == 0):
            cleanup_memory()

        if embeddings and (len(embeddings) % args.batch_save_every == 0):
            payload = {
                "embeddings": torch.stack(embeddings),
                "labels": torch.tensor(labels, dtype=torch.long),
                "label_names": LABELS,
                "sample_ids": sample_ids,
                "metadata": metadata,
                "errors": errors,
                "config": vars(args)
                | {
                    "video_max_pixels": video_max_pixels,
                    "embedding_type": f"shared_lm_{args.modality_mode}",
                    "include_video": include_video,
                    "include_utterance": include_utterance,
                },
            }
            save_payload(args.output_pt, payload)

    if not embeddings:
        raise RuntimeError(f"No embeddings were extracted. First errors: {errors[:5]}")

    print(f"Completed embeddings in output: {len(embeddings)}")
    print(f"Tracked skipped/completed sample ids: {len(done_ids)}")

    payload = {
        "embeddings": torch.stack(embeddings),
        "labels": torch.tensor(labels, dtype=torch.long),
        "label_names": LABELS,
        "sample_ids": sample_ids,
        "metadata": metadata,
        "errors": errors,
        "config": vars(args)
        | {
            "video_max_pixels": video_max_pixels,
            "embedding_type": f"shared_lm_{args.modality_mode}",
            "include_video": include_video,
            "include_utterance": include_utterance,
        },
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
