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
import time
from pathlib import Path
from typing import Any

# These must be set before importing qwen_vl_utils.
os.environ.setdefault("FORCE_QWENVL_VIDEO_READER", "decord")

import pandas as pd
import torch
from qwen_vl_utils import process_vision_info
from tqdm.auto import tqdm
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

from divprune import (
    DEFAULT_RETAIN_RATIO,
    IMPLEMENTATION_NAME as DIVPRUNE_IMPLEMENTATION,
    REFERENCE_COMMIT as DIVPRUNE_REFERENCE_COMMIT,
    build_divprune_sequence_indices,
)
from frame_diversity import (
    DEFAULT_ENCODER_MODEL as DEFAULT_FRAME_ENCODER_MODEL,
    DEFAULT_RETAIN_RATIO as DEFAULT_FRAME_RETAIN_RATIO,
    IMPLEMENTATION_NAME as FRAME_SELECTION_IMPLEMENTATION,
    FrameDiversitySelector,
)

LABELS = ["anger", "disgust", "fear", "joy", "neutral", "sadness", "surprise"]
DIRECT_ZERO_IMPLEMENTATION = "direct_placeholder_zero"
DIRECT_UTTERANCE_ZERO_IMPLEMENTATION = "direct_utterance_zero"


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
        "--frame-selection",
        choices=["none", "diverse"],
        default="none",
        help="Optionally retain a cosine-diverse subset of decoded frames before Qwen processing.",
    )
    parser.add_argument(
        "--frame-retain-ratio",
        type=float,
        default=DEFAULT_FRAME_RETAIN_RATIO,
        help="Fraction of decoded candidate frames retained by diverse frame selection (default: 0.20).",
    )
    parser.add_argument(
        "--frame-encoder-model",
        default=DEFAULT_FRAME_ENCODER_MODEL,
        help="Frozen Hugging Face image encoder used to embed candidate frames.",
    )
    parser.add_argument(
        "--frame-encoder-device",
        default="cpu",
        help="Device for the lightweight frame-selection encoder. CPU avoids competing with Qwen for VRAM.",
    )
    parser.add_argument(
        "--frame-encoder-batch-size",
        type=int,
        default=32,
        help="Candidate-frame batch size for the frame-selection encoder.",
    )
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
        "--visual-token-ablation",
        choices=["none", "zero"],
        default="none",
        help=(
            "Intervene on projected visual tokens immediately before they enter the language model. "
            "'zero' preserves token count and positions but replaces every visual token value with zero."
        ),
    )
    parser.add_argument(
        "--utterance-token-ablation",
        choices=["none", "zero"],
        default="none",
        help=(
            "Intervene on only the MELD utterance token embeddings immediately before the language model. "
            "'zero' preserves their token positions while leaving the task prompt and visual tokens unchanged."
        ),
    )
    parser.add_argument(
        "--visual-token-pruning",
        choices=["none", "divprune"],
        default="none",
        help=(
            "Optionally prune projected visual tokens immediately before the first LM decoder layer. "
            "DivPrune uses cosine-distance max-min diversity selection."
        ),
    )
    parser.add_argument(
        "--divprune-retain-ratio",
        type=float,
        default=DEFAULT_RETAIN_RATIO,
        help="Fraction of projected visual tokens retained by DivPrune (official default: 0.098).",
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
    fixed = {}
    for key, value in video_kwargs.items():
        if value == []:
            continue
        fixed[key] = value

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
    modality_mode: str,
    visual_token_ablation: str,
    utterance_token_ablation: str,
    visual_token_pruning: str,
    divprune_retain_ratio: float,
    frame_selection: str,
    frame_retain_ratio: float,
    frame_encoder_model: str,
) -> tuple[list[torch.Tensor], list[int], list[str], list[dict[str, Any]], list[dict[str, str]], set[str]]:
    if not path.exists():
        return [], [], [], [], [], set()

    payload = torch.load(path, map_location="cpu", weights_only=False)
    if "embeddings" not in payload or "labels" not in payload or "sample_ids" not in payload:
        raise ValueError(f"Cannot resume from {path}; missing embeddings, labels, or sample_ids")

    saved_config = payload.get("config", {})
    saved_modality_mode = saved_config.get("modality_mode")
    saved_ablation = saved_config.get("visual_token_ablation", "none")
    saved_utterance_ablation = saved_config.get("utterance_token_ablation", "none")
    saved_pruning = saved_config.get("visual_token_pruning", "none")
    saved_frame_selection = saved_config.get("frame_selection", "none")
    if saved_modality_mode is not None and saved_modality_mode != modality_mode:
        raise ValueError(
            f"Cannot resume from {path}; modality_mode={saved_modality_mode!r}, expected {modality_mode!r}"
        )
    if saved_ablation != visual_token_ablation:
        raise ValueError(
            f"Cannot resume from {path}; visual_token_ablation={saved_ablation!r}, "
            f"expected {visual_token_ablation!r}"
        )
    if visual_token_ablation == "zero":
        saved_implementation = saved_config.get("visual_ablation_implementation")
        if saved_implementation != DIRECT_ZERO_IMPLEMENTATION:
            raise ValueError(
                f"Cannot resume from {path}; visual_ablation_implementation={saved_implementation!r}, "
                f"expected {DIRECT_ZERO_IMPLEMENTATION!r}"
            )
    if saved_utterance_ablation != utterance_token_ablation:
        raise ValueError(
            f"Cannot resume from {path}; utterance_token_ablation={saved_utterance_ablation!r}, "
            f"expected {utterance_token_ablation!r}"
        )
    if utterance_token_ablation == "zero":
        saved_implementation = saved_config.get("utterance_ablation_implementation")
        if saved_implementation != DIRECT_UTTERANCE_ZERO_IMPLEMENTATION:
            raise ValueError(
                f"Cannot resume from {path}; utterance_ablation_implementation={saved_implementation!r}, "
                f"expected {DIRECT_UTTERANCE_ZERO_IMPLEMENTATION!r}"
            )
    if saved_pruning != visual_token_pruning:
        raise ValueError(
            f"Cannot resume from {path}; visual_token_pruning={saved_pruning!r}, "
            f"expected {visual_token_pruning!r}"
        )
    if saved_frame_selection != frame_selection:
        raise ValueError(
            f"Cannot resume from {path}; frame_selection={saved_frame_selection!r}, "
            f"expected {frame_selection!r}"
        )
    if frame_selection == "diverse":
        saved_ratio = float(saved_config.get("frame_retain_ratio", -1.0))
        if abs(saved_ratio - frame_retain_ratio) > 1e-12:
            raise ValueError(
                f"Cannot resume from {path}; frame_retain_ratio={saved_ratio}, "
                f"expected {frame_retain_ratio}"
            )
        saved_model = saved_config.get("frame_encoder_model")
        if saved_model != frame_encoder_model:
            raise ValueError(
                f"Cannot resume from {path}; frame_encoder_model={saved_model!r}, "
                f"expected {frame_encoder_model!r}"
            )
        saved_implementation = saved_config.get("frame_selection_implementation")
        if saved_implementation != FRAME_SELECTION_IMPLEMENTATION:
            raise ValueError(
                f"Cannot resume from {path}; frame_selection_implementation="
                f"{saved_implementation!r}, expected {FRAME_SELECTION_IMPLEMENTATION!r}"
            )
    if visual_token_pruning == "divprune":
        saved_ratio = float(saved_config.get("divprune_retain_ratio", -1.0))
        if abs(saved_ratio - divprune_retain_ratio) > 1e-12:
            raise ValueError(
                f"Cannot resume from {path}; divprune_retain_ratio={saved_ratio}, "
                f"expected {divprune_retain_ratio}"
            )
        saved_implementation = saved_config.get("visual_token_pruning_implementation")
        if saved_implementation != DIVPRUNE_IMPLEMENTATION:
            raise ValueError(
                f"Cannot resume from {path}; visual_token_pruning_implementation="
                f"{saved_implementation!r}, expected {DIVPRUNE_IMPLEMENTATION!r}"
            )

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


def locate_utterance_char_span(rendered_text: str, prompt: str, utterance: str) -> tuple[int, int]:
    """Locate only the utterance value inside the rendered Qwen chat template."""
    prompt_start = rendered_text.find(prompt)
    if prompt_start < 0:
        raise ValueError("Could not locate the emotion prompt inside the rendered chat template")

    marker = f"Utterance: {utterance}"
    marker_start = prompt.find(marker)
    if marker_start < 0:
        raise ValueError("Could not locate the utterance marker inside the emotion prompt")

    utterance_start = prompt_start + marker_start + len("Utterance: ")
    return utterance_start, utterance_start + len(utterance)


def align_template_tokens_to_multimodal_positions(
    template_ids: list[int],
    multimodal_ids: list[int],
    video_token_id: int,
) -> dict[int, int]:
    """Map unexpanded chat-template token indices to processor-expanded positions."""
    mapping: dict[int, int] = {}
    template_index = 0
    multimodal_index = 0

    while template_index < len(template_ids):
        template_token = template_ids[template_index]
        if template_token == video_token_id:
            if (
                multimodal_index >= len(multimodal_ids)
                or multimodal_ids[multimodal_index] != video_token_id
            ):
                raise ValueError("Video placeholder is missing from the processor-expanded input_ids")
            while (
                multimodal_index < len(multimodal_ids)
                and multimodal_ids[multimodal_index] == video_token_id
            ):
                multimodal_index += 1
            template_index += 1
            continue

        if (
            multimodal_index >= len(multimodal_ids)
            or multimodal_ids[multimodal_index] != template_token
        ):
            raise ValueError(
                "Tokenizer input_ids could not be aligned with processor input_ids at "
                f"template_index={template_index}, multimodal_index={multimodal_index}"
            )
        mapping[template_index] = multimodal_index
        template_index += 1
        multimodal_index += 1

    if multimodal_index != len(multimodal_ids):
        raise ValueError(
            "Processor input_ids contain unmatched trailing tokens: "
            f"matched={multimodal_index}, total={len(multimodal_ids)}"
        )
    return mapping


def build_utterance_token_mask(
    processor: AutoProcessor,
    rendered_text: str,
    prompt: str,
    utterance: str,
    processor_input_ids: torch.Tensor,
    video_token_id: int,
) -> torch.Tensor:
    """Return a boolean mask for only the utterance tokens in the expanded sequence."""
    if processor_input_ids.ndim != 2 or processor_input_ids.shape[0] != 1:
        raise ValueError(
            "Utterance-token ablation expects input_ids shaped [1, sequence_length], "
            f"got {tuple(processor_input_ids.shape)}"
        )
    if not utterance:
        raise ValueError("Utterance-token ablation requires a non-empty utterance")

    utterance_start, utterance_end = locate_utterance_char_span(rendered_text, prompt, utterance)
    tokenized = processor.tokenizer(
        [rendered_text],
        padding=True,
        return_tensors="pt",
        return_offsets_mapping=True,
    )
    template_ids = tokenized["input_ids"][0].tolist()
    offsets = tokenized["offset_mapping"][0].tolist()
    multimodal_ids = processor_input_ids[0].detach().cpu().tolist()
    position_map = align_template_tokens_to_multimodal_positions(
        template_ids,
        multimodal_ids,
        video_token_id,
    )

    template_utterance_indices = [
        index
        for index, (start, end) in enumerate(offsets)
        if end > utterance_start and start < utterance_end
    ]
    if not template_utterance_indices:
        raise ValueError("Tokenizer offsets did not identify any utterance tokens")

    mask = torch.zeros_like(processor_input_ids, dtype=torch.bool)
    for template_index in template_utterance_indices:
        multimodal_position = position_map.get(template_index)
        if multimodal_position is None:
            raise ValueError("An utterance token unexpectedly aligned to a video placeholder")
        mask[0, multimodal_position] = True
    return mask


def forward_multimodal_hidden_state(
    model: Qwen2_5_VLForConditionalGeneration,
    inputs: dict[str, torch.Tensor],
    layer: int,
    visual_token_ablation: str,
    utterance_token_ablation: str,
    utterance_token_mask: torch.Tensor | None,
    visual_token_pruning: str,
    divprune_retain_ratio: float,
) -> tuple[torch.Tensor, torch.Tensor, int]:
    """Return one shared transformer hidden-state tensor before Qwen's LM head."""
    base_model = getattr(model, "model", None)
    utterance_hook = None
    pruning_hook = None
    utterance_hook_calls = 0
    pruning_hook_calls = 0
    effective_attention_mask = inputs["attention_mask"]
    retained_visual_token_count = 0

    input_ids = inputs.get("input_ids")
    video_token_id = getattr(getattr(base_model, "config", None), "video_token_id", None)
    video_mask = None
    if input_ids is not None and video_token_id is not None:
        video_mask = input_ids.eq(video_token_id)
        retained_visual_token_count = int(video_mask.sum().item())

    if utterance_token_ablation == "zero":
        if base_model is None:
            raise AttributeError("Utterance-token ablation requires the Qwen base model")
        if utterance_token_mask is None or not utterance_token_mask.any():
            raise ValueError("Utterance-token ablation requires a non-empty utterance token mask")
        language_model = getattr(base_model, "language_model", None)
        if language_model is None:
            raise AttributeError("Could not find Qwen language_model for utterance-token ablation")

        def zero_utterance_embeddings(
            _module: torch.nn.Module,
            module_args: tuple[Any, ...],
            module_kwargs: dict[str, Any],
        ) -> tuple[tuple[Any, ...], dict[str, Any]]:
            nonlocal utterance_hook_calls
            inputs_embeds = module_kwargs.get("inputs_embeds")
            if not isinstance(inputs_embeds, torch.Tensor):
                raise TypeError("Qwen language_model did not receive tensor inputs_embeds")
            mask = utterance_token_mask.to(device=inputs_embeds.device).unsqueeze(-1)
            if mask.shape[:2] != inputs_embeds.shape[:2]:
                raise ValueError(
                    "Utterance mask and LM embeddings have different sequence shapes: "
                    f"{tuple(mask.shape[:2])} vs {tuple(inputs_embeds.shape[:2])}"
                )
            updated_kwargs = dict(module_kwargs)
            updated_kwargs["inputs_embeds"] = inputs_embeds.masked_fill(mask, 0.0)
            utterance_hook_calls += 1
            return module_args, updated_kwargs

        utterance_hook = language_model.register_forward_pre_hook(
            zero_utterance_embeddings,
            with_kwargs=True,
        )

    if visual_token_pruning == "divprune":
        if base_model is None:
            raise AttributeError("DivPrune requires the Qwen base model")
        language_model = getattr(base_model, "language_model", None)
        if language_model is None:
            raise AttributeError("Could not find Qwen language_model for DivPrune")
        if video_mask is None or not video_mask.any():
            raise ValueError("DivPrune requires video placeholder positions in input_ids")
        if video_mask.ndim != 2 or video_mask.shape[0] != 1:
            raise ValueError(
                f"DivPrune currently expects one sample per forward pass, got {tuple(video_mask.shape)}"
            )

        def prune_projected_visual_tokens(
            _module: torch.nn.Module,
            module_args: tuple[Any, ...],
            module_kwargs: dict[str, Any],
        ) -> tuple[tuple[Any, ...], dict[str, Any]]:
            nonlocal pruning_hook_calls, effective_attention_mask, retained_visual_token_count
            inputs_embeds = module_kwargs.get("inputs_embeds")
            if not isinstance(inputs_embeds, torch.Tensor):
                raise TypeError("Qwen language_model did not receive tensor inputs_embeds")
            if inputs_embeds.ndim != 3 or inputs_embeds.shape[0] != 1:
                raise ValueError(
                    "DivPrune expects LM inputs_embeds shaped [1, sequence_length, hidden_dim], "
                    f"got {tuple(inputs_embeds.shape)}"
                )

            current_video_mask = video_mask.to(device=inputs_embeds.device)
            if current_video_mask.shape != inputs_embeds.shape[:2]:
                raise ValueError(
                    "Video placeholder mask and LM embeddings have different sequence shapes: "
                    f"{tuple(current_video_mask.shape)} vs {tuple(inputs_embeds.shape[:2])}"
                )
            keep_indices, selected_visual_positions = build_divprune_sequence_indices(
                inputs_embeds,
                current_video_mask,
                retain_ratio=divprune_retain_ratio,
            )

            updated_kwargs = dict(module_kwargs)
            updated_kwargs["inputs_embeds"] = inputs_embeds.index_select(1, keep_indices)

            attention_mask = module_kwargs.get("attention_mask")
            if not isinstance(attention_mask, torch.Tensor):
                raise TypeError("DivPrune requires a tensor attention_mask")
            if attention_mask.shape[-1] != inputs_embeds.shape[1]:
                raise ValueError("Attention mask length does not match the unpruned LM sequence")
            updated_attention_mask = attention_mask.index_select(
                attention_mask.ndim - 1,
                keep_indices.to(attention_mask.device),
            )
            updated_kwargs["attention_mask"] = updated_attention_mask

            position_ids = module_kwargs.get("position_ids")
            if not isinstance(position_ids, torch.Tensor):
                raise TypeError("DivPrune requires Qwen multimodal position_ids")
            if position_ids.shape[-1] != inputs_embeds.shape[1]:
                raise ValueError("Position ID length does not match the unpruned LM sequence")
            updated_kwargs["position_ids"] = position_ids.index_select(
                position_ids.ndim - 1,
                keep_indices.to(position_ids.device),
            )

            cache_position = module_kwargs.get("cache_position")
            if (
                isinstance(cache_position, torch.Tensor)
                and cache_position.shape[-1] == inputs_embeds.shape[1]
            ):
                updated_kwargs["cache_position"] = cache_position.index_select(
                    cache_position.ndim - 1,
                    keep_indices.to(cache_position.device),
                )

            effective_attention_mask = updated_attention_mask
            retained_visual_token_count = int(selected_visual_positions.numel())
            pruning_hook_calls += 1
            return module_args, updated_kwargs

        pruning_hook = language_model.register_forward_pre_hook(
            prune_projected_visual_tokens,
            with_kwargs=True,
        )

    try:
        if visual_token_ablation == "zero":
            if base_model is None:
                raise AttributeError("Zero-token ablation requires the Qwen base model")

            input_ids = inputs.get("input_ids")
            if input_ids is None:
                raise ValueError("Zero-token ablation requires input_ids to locate video placeholders")

            video_token_id = getattr(base_model.config, "video_token_id", None)
            if video_token_id is None:
                raise AttributeError("Qwen config does not expose video_token_id")

            video_mask = input_ids.eq(video_token_id)
            video_token_count = int(video_mask.sum().item())
            if video_token_count == 0:
                raise ValueError("No video placeholder tokens were found in input_ids")

            video_grid_thw = inputs.get("video_grid_thw")
            if video_grid_thw is None:
                raise ValueError("Zero-token ablation requires video_grid_thw for multimodal position IDs")
            spatial_merge_size = int(base_model.config.vision_config.spatial_merge_size)
            expected_video_tokens = int(
                (video_grid_thw.prod(dim=-1) // (spatial_merge_size**2)).sum().item()
            )
            if video_token_count != expected_video_tokens:
                raise ValueError(
                    "Video placeholder count does not match the processor grid: "
                    f"placeholders={video_token_count}, expected={expected_video_tokens}"
                )

            inputs_embeds = base_model.get_input_embeddings()(input_ids)
            inputs_embeds = inputs_embeds.masked_fill(video_mask.unsqueeze(-1), 0.0)

            # Keep input_ids and grid metadata so Qwen computes its original 3D multimodal
            # RoPE positions, but omit pixels so the visual encoder is never called.
            ablated_inputs = {
                key: value
                for key, value in inputs.items()
                if key not in {"pixel_values_videos", "pixel_values"}
            }
            ablated_inputs["inputs_embeds"] = inputs_embeds
            outputs = base_model(
                **ablated_inputs,
                output_hidden_states=(layer != -1),
                use_cache=False,
                return_dict=True,
            )
            hidden = outputs.last_hidden_state if layer == -1 else outputs.hidden_states[layer]
        elif base_model is not None:
            outputs = base_model(
                **inputs,
                output_hidden_states=(layer != -1),
                use_cache=False,
                return_dict=True,
            )
            hidden = outputs.last_hidden_state if layer == -1 else outputs.hidden_states[layer]
        else:
            if utterance_token_ablation != "none":
                raise AttributeError("Utterance-token ablation is unavailable for this model wrapper")
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
            hidden = outputs.hidden_states[layer]
    finally:
        if pruning_hook is not None:
            pruning_hook.remove()
        if utterance_hook is not None:
            utterance_hook.remove()

    if utterance_token_ablation == "zero" and utterance_hook_calls != 1:
        raise RuntimeError(
            "Utterance-token ablation expected one language-model call, "
            f"observed {utterance_hook_calls}"
        )
    if visual_token_pruning == "divprune" and pruning_hook_calls != 1:
        raise RuntimeError(
            "DivPrune expected one language-model call, "
            f"observed {pruning_hook_calls}"
        )
    return hidden, effective_attention_mask, retained_visual_token_count


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
    visual_token_ablation: str,
    utterance_token_ablation: str,
    visual_token_pruning: str,
    divprune_retain_ratio: float,
    frame_selector: FrameDiversitySelector | None,
    frame_retain_ratio: float,
) -> tuple[torch.Tensor, int, int, int, dict[str, Any]]:
    prompt = make_prompt(utterance, prompt_style, modality_mode)
    messages = make_message(video_path, utterance, fps, min_frames, max_frames, frame_size, prompt_style, modality_mode)
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=add_generation_prompt)
    image_inputs, video_inputs, video_kwargs = process_vision_info(messages, return_video_kwargs=True)
    video_kwargs = fix_video_kwargs(video_kwargs)

    frame_selection_metadata: dict[str, Any] = {
        "candidate_frame_count": None,
        "retained_frame_count": None,
        "actual_frame_retain_ratio": None,
        "selected_frame_indices": None,
        "frame_selection_seconds": None,
        "candidate_sampling_fps": None,
        "qwen_input_fps": None,
    }
    if frame_selector is not None:
        if not video_inputs or len(video_inputs) != 1:
            raise ValueError(
                "Diverse frame selection expects exactly one decoded video per sample"
            )
        candidate_frames = video_inputs[0]
        if not isinstance(candidate_frames, torch.Tensor) or candidate_frames.ndim != 4:
            raise ValueError(
                "Diverse frame selection expected a decoded TCHW video tensor, "
                f"got {type(candidate_frames).__name__}"
            )
        selection_start = time.perf_counter()
        selected_frames, selected_indices = frame_selector.select(
            candidate_frames,
            retain_ratio=frame_retain_ratio,
        )
        selection_seconds = time.perf_counter() - selection_start
        candidate_count = int(candidate_frames.shape[0])
        retained_count = int(selected_frames.shape[0])
        video_inputs = list(video_inputs)
        video_inputs[0] = selected_frames

        # Qwen uses fps to construct temporal position intervals. The retained
        # frames still span the original clip, so preserve its average duration.
        candidate_sampling_fps = (
            float(video_kwargs["fps"]) if "fps" in video_kwargs else None
        )
        if candidate_sampling_fps is not None:
            video_kwargs["fps"] = candidate_sampling_fps * retained_count / candidate_count

        frame_selection_metadata = {
            "candidate_frame_count": candidate_count,
            "retained_frame_count": retained_count,
            "actual_frame_retain_ratio": retained_count / candidate_count,
            "selected_frame_indices": selected_indices.tolist(),
            "frame_selection_seconds": selection_seconds,
            "candidate_sampling_fps": candidate_sampling_fps,
            "qwen_input_fps": video_kwargs.get("fps"),
        }

    processor_inputs: dict[str, Any] = {
        "text": [text],
        "padding": True,
        "return_tensors": "pt",
    }
    if image_inputs:
        processor_inputs["images"] = image_inputs
    if video_inputs:
        processor_inputs["videos"] = video_inputs
    processor_inputs.update(video_kwargs)

    inputs = processor(**processor_inputs)
    base_model = getattr(model, "model", None)
    video_token_id = getattr(getattr(base_model, "config", None), "video_token_id", None)
    if video_token_id is None:
        raise AttributeError("Qwen config does not expose video_token_id")

    utterance_token_mask = None
    if utterance_token_ablation == "zero":
        utterance_token_mask = build_utterance_token_mask(
            processor=processor,
            rendered_text=text,
            prompt=prompt,
            utterance="" if utterance is None else utterance,
            processor_input_ids=inputs["input_ids"],
            video_token_id=video_token_id,
        )
    utterance_token_count = (
        int(utterance_token_mask.sum().item()) if utterance_token_mask is not None else 0
    )

    if visual_token_ablation == "zero":
        # The processor is still the source of truth for placeholder expansion and
        # grid metadata, but its pixel tensor is unnecessary for this intervention.
        inputs.pop("pixel_values_videos", None)
        inputs.pop("pixel_values", None)
    inputs = inputs.to(model.device)
    video_token_count = (
        int(inputs["input_ids"].eq(video_token_id).sum().item())
        if video_token_id is not None and "input_ids" in inputs
        else 0
    )

    try:
        with torch.inference_mode():
            (
                hidden,
                effective_attention_mask,
                retained_video_token_count,
            ) = forward_multimodal_hidden_state(
                model,
                inputs,
                layer,
                visual_token_ablation,
                utterance_token_ablation,
                utterance_token_mask,
                visual_token_pruning,
                divprune_retain_ratio,
            )

        embedding = pool_hidden_states(
            hidden.detach().float().cpu(),
            effective_attention_mask.detach().cpu(),
            pooling,
        )
    finally:
        del prompt, messages, text, image_inputs, video_inputs, video_kwargs, processor_inputs, inputs
        if "hidden" in locals():
            del hidden
        cleanup_memory()

    return (
        embedding,
        video_token_count,
        retained_video_token_count,
        utterance_token_count,
        frame_selection_metadata,
    )


def save_payload(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, tmp_path)
    tmp_path.replace(path)


def main() -> None:
    args = parse_args()

    if args.visual_token_ablation != "none" and args.modality_mode == "text_only":
        raise ValueError("Visual-token ablation requires a modality mode that includes video")
    if args.utterance_token_ablation != "none" and args.modality_mode == "video_only":
        raise ValueError("Utterance-token ablation requires a modality mode that includes the utterance")
    if args.visual_token_pruning != "none" and args.modality_mode == "text_only":
        raise ValueError("Visual-token pruning requires a modality mode that includes video")
    if args.visual_token_pruning != "none" and args.visual_token_ablation != "none":
        raise ValueError("Visual-token pruning and visual-token ablation cannot be combined")
    if not 0.0 < args.divprune_retain_ratio <= 1.0:
        raise ValueError("--divprune-retain-ratio must be in the interval (0, 1]")
    if args.frame_selection != "none" and args.modality_mode == "text_only":
        raise ValueError("Frame selection requires a modality mode that includes video")
    if not 0.0 < args.frame_retain_ratio <= 1.0:
        raise ValueError("--frame-retain-ratio must be in the interval (0, 1]")
    if args.frame_encoder_batch_size <= 0:
        raise ValueError("--frame-encoder-batch-size must be positive")

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
    print(
        f"Frame selection: {args.frame_selection}; retain ratio={args.frame_retain_ratio}; "
        f"encoder={args.frame_encoder_model}"
    )
    print(f"Visual-token ablation: {args.visual_token_ablation}")
    print(f"Utterance-token ablation: {args.utterance_token_ablation}")
    print(
        f"Visual-token pruning: {args.visual_token_pruning}; "
        f"DivPrune retain ratio={args.divprune_retain_ratio}"
    )
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

    frame_selector = None
    if args.frame_selection == "diverse":
        frame_selector = FrameDiversitySelector(
            model_id=args.frame_encoder_model,
            device=args.frame_encoder_device,
            batch_size=args.frame_encoder_batch_size,
            trust_remote_code=args.trust_remote_code,
        )

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(args.model_id, **model_kwargs)
    processor = AutoProcessor.from_pretrained(args.model_id, **processor_kwargs)
    model.eval()

    output_dtype = save_embedding_dtype(args.save_dtype)

    if args.resume:
        embeddings, labels, sample_ids, metadata, errors, done_ids = load_resume_payload(
            args.output_pt,
            output_dtype=output_dtype,
            retry_errors=args.retry_errors,
            modality_mode=args.modality_mode,
            visual_token_ablation=args.visual_token_ablation,
            utterance_token_ablation=args.utterance_token_ablation,
            visual_token_pruning=args.visual_token_pruning,
            divprune_retain_ratio=args.divprune_retain_ratio,
            frame_selection=args.frame_selection,
            frame_retain_ratio=args.frame_retain_ratio,
            frame_encoder_model=args.frame_encoder_model,
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
            (
                embedding,
                video_token_count,
                retained_video_token_count,
                utterance_token_count,
                frame_selection_metadata,
            ) = extract_shared_embedding(
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
                visual_token_ablation=args.visual_token_ablation,
                utterance_token_ablation=args.utterance_token_ablation,
                visual_token_pruning=args.visual_token_pruning,
                divprune_retain_ratio=args.divprune_retain_ratio,
                frame_selector=frame_selector,
                frame_retain_ratio=args.frame_retain_ratio,
            )
            if not torch.isfinite(embedding).all():
                raise ValueError("Extracted embedding contains NaN or Inf values")
            embeddings.append(embedding.to(dtype=output_dtype))
            labels.append(int(row["emotion_id"]))
            sample_ids.append(sample_id)
            row_metadata = row.to_dict()
            row_metadata["modality_mode"] = args.modality_mode
            row_metadata["include_video"] = include_video
            row_metadata["include_utterance"] = include_utterance
            row_metadata["visual_token_ablation"] = args.visual_token_ablation
            row_metadata["visual_ablation_implementation"] = (
                DIRECT_ZERO_IMPLEMENTATION if args.visual_token_ablation == "zero" else "none"
            )
            row_metadata["utterance_token_ablation"] = args.utterance_token_ablation
            row_metadata["utterance_ablation_implementation"] = (
                DIRECT_UTTERANCE_ZERO_IMPLEMENTATION
                if args.utterance_token_ablation == "zero"
                else "none"
            )
            row_metadata["video_token_count"] = video_token_count
            row_metadata["original_visual_token_count"] = video_token_count
            row_metadata["retained_visual_token_count"] = retained_video_token_count
            row_metadata["pruned_visual_token_count"] = video_token_count - retained_video_token_count
            row_metadata["actual_visual_token_retain_ratio"] = (
                retained_video_token_count / video_token_count if video_token_count else None
            )
            row_metadata["visual_token_pruning"] = args.visual_token_pruning
            row_metadata["visual_token_pruning_implementation"] = (
                DIVPRUNE_IMPLEMENTATION if args.visual_token_pruning == "divprune" else "none"
            )
            row_metadata["divprune_reference_commit"] = (
                DIVPRUNE_REFERENCE_COMMIT if args.visual_token_pruning == "divprune" else None
            )
            row_metadata["utterance_token_count"] = utterance_token_count
            row_metadata.update(frame_selection_metadata)
            row_metadata["frame_selection"] = args.frame_selection
            row_metadata["frame_selection_implementation"] = (
                FRAME_SELECTION_IMPLEMENTATION if args.frame_selection == "diverse" else "none"
            )
            row_metadata["frame_encoder_model"] = (
                args.frame_encoder_model if args.frame_selection == "diverse" else None
            )
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
                    "embedding_type": (
                        f"shared_lm_{args.modality_mode}"
                        + ("_zero_visual_tokens" if args.visual_token_ablation == "zero" else "")
                        + ("_zero_utterance_tokens" if args.utterance_token_ablation == "zero" else "")
                        + ("_divprune" if args.visual_token_pruning == "divprune" else "")
                        + ("_diverse_frames_020" if args.frame_selection == "diverse" else "")
                    ),
                    "visual_ablation_implementation": (
                        DIRECT_ZERO_IMPLEMENTATION if args.visual_token_ablation == "zero" else "none"
                    ),
                    "utterance_ablation_implementation": (
                        DIRECT_UTTERANCE_ZERO_IMPLEMENTATION
                        if args.utterance_token_ablation == "zero"
                        else "none"
                    ),
                    "visual_token_pruning_implementation": (
                        DIVPRUNE_IMPLEMENTATION if args.visual_token_pruning == "divprune" else "none"
                    ),
                    "divprune_reference_commit": (
                        DIVPRUNE_REFERENCE_COMMIT if args.visual_token_pruning == "divprune" else None
                    ),
                    "frame_selection_implementation": (
                        FRAME_SELECTION_IMPLEMENTATION if args.frame_selection == "diverse" else "none"
                    ),
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
            "embedding_type": (
                f"shared_lm_{args.modality_mode}"
                + ("_zero_visual_tokens" if args.visual_token_ablation == "zero" else "")
                + ("_zero_utterance_tokens" if args.utterance_token_ablation == "zero" else "")
                + ("_divprune" if args.visual_token_pruning == "divprune" else "")
                + ("_diverse_frames_020" if args.frame_selection == "diverse" else "")
            ),
            "visual_ablation_implementation": (
                DIRECT_ZERO_IMPLEMENTATION if args.visual_token_ablation == "zero" else "none"
            ),
            "utterance_ablation_implementation": (
                DIRECT_UTTERANCE_ZERO_IMPLEMENTATION
                if args.utterance_token_ablation == "zero"
                else "none"
            ),
            "visual_token_pruning_implementation": (
                DIVPRUNE_IMPLEMENTATION if args.visual_token_pruning == "divprune" else "none"
            ),
            "divprune_reference_commit": (
                DIVPRUNE_REFERENCE_COMMIT if args.visual_token_pruning == "divprune" else None
            ),
            "frame_selection_implementation": (
                FRAME_SELECTION_IMPLEMENTATION if args.frame_selection == "diverse" else "none"
            ),
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
