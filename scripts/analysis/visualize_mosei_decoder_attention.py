#!/usr/bin/env python3
"""Visualize Qwen decoder attention from the final prompt token on CMU-MOSEI."""

from __future__ import annotations

# Allow direct execution from a read-only Kaggle code dataset.
import sys as _sys
from pathlib import Path as _Path
if __package__ in (None, ""):
    _sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))

import argparse
import csv
import gc
import hashlib
import json
import math
import os
import re
from pathlib import Path
from typing import Any

import numpy as np

from scripts.analysis.qwen_decoder_attention import QwenLastQueryAttentionCapture


EMOTIONS = ("happy", "sad", "anger", "surprise", "disgust", "fear")


def make_message(video_path: str, utterance: str, fps: float, min_frames: int,
                 max_frames: int, frame_size: int) -> list[dict[str, Any]]:
    """Build the same video-and-utterance message as the MOSEI baseline."""
    return [{"role": "user", "content": [
        {
            "type": "video",
            "video": video_path,
            "fps": fps,
            "min_frames": min_frames,
            "max_frames": max_frames,
            "resized_height": frame_size,
            "resized_width": frame_size,
        },
        {"type": "text", "text": f"Utterance: {utterance}"},
    ]}]


def fix_video_kwargs(video_kwargs: dict[str, Any]) -> dict[str, Any]:
    """Handle qwen-vl-utils versions that return one-video FPS as a list."""
    fixed = {key: value for key, value in video_kwargs.items() if value != []}
    if "fps" in fixed and isinstance(fixed["fps"], list) and len(fixed["fps"]) == 1:
        fixed["fps"] = fixed["fps"][0]
    return fixed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model-id", default="Qwen/Qwen2.5-VL-3B-Instruct")
    parser.add_argument("--sample-id", action="append", default=[],
                        help="Exact sample ID to analyze; repeat for multiple clips")
    parser.add_argument("--num-clips", type=int, default=3,
                        help="Use the first N index rows when --sample-id is omitted")
    parser.add_argument("--fps", type=float, default=6.0)
    parser.add_argument("--min-frames", type=int, default=4)
    parser.add_argument("--max-frames", type=int, default=64)
    parser.add_argument("--frame-size", type=int, default=224)
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--skip-existing", action="store_true")
    return parser.parse_args()


def safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_") or "sample"


def completed_sample(sample_dir: Path, args, sample_id: str) -> bool:
    """Return true only when a prior clip has every expected, matching output."""
    try:
        metadata = json.loads((sample_dir / "metadata.json").read_text())
        expected = {
            "sample_id": sample_id,
            "model_id": args.model_id,
            "analyzer_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "fps": args.fps,
            "min_frames": args.min_frames,
            "max_frames": args.max_frames,
            "frame_size": args.frame_size,
            "head_reduction": "arithmetic mean over all language-decoder attention heads",
        }
        if any(metadata.get(key) != value for key, value in expected.items()):
            return False
        required = (
            "attention_arrays.npz", "layer_summary.csv", "visual_token_attention.csv",
            "modality_attention_by_layer.png", "temporal_attention_by_layer.png",
            "spatial_attention_all_layers.png",
        )
        if any(not (sample_dir / name).is_file() for name in required):
            return False
        with np.load(sample_dir / "attention_arrays.npz") as arrays:
            mean_attention = arrays["mean_attention"]
            video_attention = arrays["video_attention"]
        layers = int(metadata["attention_layers"])
        sequence = int(metadata["sequence_tokens"])
        if mean_attention.shape != (layers, sequence) or video_attention.shape[0] != layers:
            return False
        overlays = list((sample_dir / "layer_overlays").glob("layer_*.png"))
        return len(overlays) == layers
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError):
        return False


def load_rows(path: Path, requested: list[str], count: int) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"No rows found in {path}")
    required = {"sample_id", "video_path", "utterance"}
    missing = required - set(rows[0])
    if missing:
        raise ValueError(f"Index is missing columns: {sorted(missing)}")
    if requested:
        by_id = {row["sample_id"]: row for row in rows}
        absent = [sample_id for sample_id in requested if sample_id not in by_id]
        if absent:
            raise ValueError(f"Requested sample IDs are absent from the index: {absent}")
        return [by_id[sample_id] for sample_id in requested]
    if count <= 0:
        raise ValueError("--num-clips must be positive")
    return rows[:count]


def rgb_frames(video_item) -> np.ndarray:
    """Convert qwen-vl-utils video output to uint8 ``[frames, height, width, 3]``."""
    import torch

    frames = video_item[0] if isinstance(video_item, tuple) else video_item
    if torch.is_tensor(frames):
        frames = frames.detach().cpu().numpy()
    frames = np.asarray(frames)
    if frames.ndim != 4:
        raise ValueError(f"Expected four-dimensional video frames, got {frames.shape}")
    if frames.shape[1] in (1, 3) and frames.shape[-1] not in (1, 3):
        frames = np.moveaxis(frames, 1, -1)
    if frames.shape[-1] == 1:
        frames = np.repeat(frames, 3, axis=-1)
    if frames.shape[-1] != 3:
        raise ValueError(f"Expected RGB frames, got {frames.shape}")
    if frames.dtype != np.uint8:
        if np.issubdtype(frames.dtype, np.floating) and frames.max(initial=0) <= 1.0:
            frames = frames * 255.0
        frames = np.clip(frames, 0, 255).astype(np.uint8)
    return frames


def save_layer_summary(path: Path, attention: np.ndarray, video_mask: np.ndarray) -> None:
    visual = np.flatnonzero(video_mask)
    first, last = int(visual[0]), int(visual[-1])
    fields = (
        "layer", "prefix_mass", "video_mass", "post_video_mass",
        "query_self_attention", "attention_entropy",
    )
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for layer, scores in enumerate(attention):
            writer.writerow({
                "layer": layer,
                "prefix_mass": float(scores[:first].sum()),
                "video_mass": float(scores[video_mask].sum()),
                "post_video_mass": float(scores[last + 1:].sum()),
                "query_self_attention": float(scores[-1]),
                "attention_entropy": float(-(scores * np.log(scores + 1e-30)).sum()),
            })


def save_visual_scores(path: Path, visual: np.ndarray) -> None:
    layers, temporal, rows, columns = visual.shape
    fields = ("layer", "temporal_group", "row", "column", "attention")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for layer in range(layers):
            for group in range(temporal):
                for row in range(rows):
                    for column in range(columns):
                        writer.writerow({
                            "layer": layer,
                            "temporal_group": group,
                            "row": row,
                            "column": column,
                            "attention": float(visual[layer, group, row, column]),
                        })


def plot_summaries(output: Path, attention: np.ndarray, video_mask: np.ndarray,
                   visual: np.ndarray) -> None:
    import matplotlib.pyplot as plt

    layers = np.arange(attention.shape[0])
    positions = np.flatnonzero(video_mask)
    first, last = int(positions[0]), int(positions[-1])
    prefix = attention[:, :first].sum(axis=1)
    video = attention[:, video_mask].sum(axis=1)
    suffix = attention[:, last + 1:].sum(axis=1)

    fig, axis = plt.subplots(figsize=(9, 4.8))
    axis.plot(layers, prefix, marker="o", markersize=3, label="Before video")
    axis.plot(layers, video, marker="o", markersize=3, label="Video tokens")
    axis.plot(layers, suffix, marker="o", markersize=3, label="After video")
    axis.set(xlabel="Language-decoder layer (zero-based)", ylabel="Attention mass",
             title="Final prompt-token attention by sequence region")
    axis.set_ylim(bottom=0)
    axis.grid(alpha=0.25)
    axis.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output / "modality_attention_by_layer.png", dpi=180)
    plt.close(fig)

    temporal = visual.sum(axis=(2, 3))
    conditional = temporal / np.maximum(temporal.sum(axis=1, keepdims=True), 1e-30)
    fig, axis = plt.subplots(figsize=(max(8, conditional.shape[1] * 0.24), 6))
    image = axis.imshow(conditional, aspect="auto", interpolation="nearest", cmap="magma")
    axis.set(xlabel="Temporal visual-token group", ylabel="Decoder layer",
             title="Within-video temporal attention (head mean, row-normalized)")
    fig.colorbar(image, ax=axis, label="Share of video attention")
    fig.tight_layout()
    fig.savefig(output / "temporal_attention_by_layer.png", dpi=180)
    plt.close(fig)

    count = visual.shape[0]
    columns = min(6, count)
    rows = math.ceil(count / columns)
    fig, axes = plt.subplots(rows, columns, figsize=(2.4 * columns, 2.2 * rows), squeeze=False)
    for layer, axis in enumerate(axes.flat):
        if layer >= count:
            axis.axis("off")
            continue
        spatial = visual[layer].sum(axis=0)
        spatial = spatial / max(float(spatial.sum()), 1e-30)
        axis.imshow(spatial, cmap="magma", interpolation="nearest")
        axis.set_title(f"Layer {layer}", fontsize=9)
        axis.set_xticks([])
        axis.set_yticks([])
    fig.suptitle("Within-video spatial attention, summed over time", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    fig.savefig(output / "spatial_attention_all_layers.png", dpi=170)
    plt.close(fig)


def overlay_contact_sheets(output: Path, frames: np.ndarray, visual: np.ndarray,
                           temporal_patch: int) -> None:
    import cv2

    overlay_dir = output / "layer_overlays"
    overlay_dir.mkdir(parents=True, exist_ok=True)
    temporal = visual.shape[1]
    columns = min(8, temporal)
    cell = 224
    header_height = 34
    label_height = 24
    rows = math.ceil(temporal / columns)
    global_scale = max(float(visual.max()), 1e-30)
    for layer in range(visual.shape[0]):
        layer_scores = visual[layer]
        canvas = np.full(
            (header_height + rows * (cell + label_height), columns * cell, 3),
            245, dtype=np.uint8,
        )
        title = (
            f"Layer {layer:02d} | raw video mass={layer_scores.sum():.6f} | "
            f"token max={layer_scores.max():.3e} | shared max={global_scale:.3e}"
        )
        cv2.putText(canvas, title, (8, 23), cv2.FONT_HERSHEY_SIMPLEX,
                    0.55, (25, 25, 25), 1, cv2.LINE_AA)
        for group in range(temporal):
            frame_index = min(group * temporal_patch, len(frames) - 1)
            frame = cv2.resize(frames[frame_index], (cell, cell), interpolation=cv2.INTER_AREA)
            heat = cv2.resize(
                (layer_scores[group] / global_scale).astype(np.float32),
                (cell, cell), interpolation=cv2.INTER_CUBIC,
            )
            color = cv2.cvtColor(
                cv2.applyColorMap(np.uint8(np.clip(heat, 0, 1) * 255), cv2.COLORMAP_INFERNO),
                cv2.COLOR_BGR2RGB,
            )
            alpha = (0.70 * np.clip(heat, 0, 1))[..., None]
            blended = np.uint8(np.clip(frame * (1 - alpha) + color * alpha, 0, 255))
            row, column = divmod(group, columns)
            y0, x0 = header_height + row * (cell + label_height), column * cell
            canvas[y0:y0 + cell, x0:x0 + cell] = blended
            group_mass = float(layer_scores[group].sum())
            cv2.putText(canvas, f"t={group} frame={frame_index} mass={group_mass:.2e}",
                        (x0 + 5, y0 + cell + 17), cv2.FONT_HERSHEY_SIMPLEX,
                        0.36, (30, 30, 30), 1, cv2.LINE_AA)
        bgr = cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR)
        cv2.imwrite(str(overlay_dir / f"layer_{layer:02d}.png"), bgr)


def analyze_sample(args, row, model, processor) -> dict:
    import torch
    from qwen_vl_utils import process_vision_info

    sample_id = row["sample_id"]
    sample_dir = args.output_dir / safe_name(sample_id)
    if args.skip_existing and completed_sample(sample_dir, args, sample_id):
        print(f"Skipping completed sample {sample_id}", flush=True)
        return json.loads((sample_dir / "metadata.json").read_text())
    sample_dir.mkdir(parents=True, exist_ok=True)
    video_path = Path(row["video_path"])
    if not video_path.is_file():
        raise FileNotFoundError(video_path)

    messages = make_message(
        str(video_path), row["utterance"], args.fps, args.min_frames,
        args.max_frames, args.frame_size,
    )
    prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs, video_kwargs = process_vision_info(
        messages, return_video_kwargs=True
    )
    frames = rgb_frames(video_inputs[0])
    inputs = processor(
        text=[prompt], images=image_inputs, videos=video_inputs, padding=True,
        return_tensors="pt", **fix_video_kwargs(video_kwargs),
    ).to(model.device)
    sequence_length = int(inputs["input_ids"].shape[-1])
    with torch.inference_mode(), QwenLastQueryAttentionCapture(
        model, sequence_length
    ) as capture:
        output = model.model(**inputs, use_cache=False, return_dict=True)
    attention = capture.stacked().numpy()
    del output

    input_ids = inputs["input_ids"][0].detach().cpu()
    video_mask = input_ids.eq(model.config.video_token_id).numpy()
    if not video_mask.any():
        raise ValueError("The processed sequence contains no video tokens")
    grid = tuple(map(int, inputs["video_grid_thw"][0].detach().cpu().tolist()))
    merge = int(model.config.vision_config.spatial_merge_size)
    temporal_patch = int(model.config.vision_config.temporal_patch_size)
    temporal, grid_height, grid_width = grid
    if grid_height % merge or grid_width % merge:
        raise ValueError(f"Video grid {grid} is incompatible with spatial merge {merge}")
    rows, columns = grid_height // merge, grid_width // merge
    expected_visual = temporal * rows * columns
    if int(video_mask.sum()) != expected_visual:
        raise ValueError(
            f"Prompt has {int(video_mask.sum())} video tokens; grid implies {expected_visual}"
        )
    if len(frames) != temporal * temporal_patch:
        raise ValueError(
            f"Grid and temporal patching imply {temporal * temporal_patch} frames; "
            f"decoder returned {len(frames)}"
        )
    visual = attention[:, video_mask].reshape(attention.shape[0], temporal, rows, columns)

    np.savez_compressed(
        sample_dir / "attention_arrays.npz",
        mean_attention=attention,
        video_attention=visual,
        input_ids=input_ids.numpy(),
        video_mask=video_mask,
        video_grid_thw=np.asarray(grid),
    )
    save_layer_summary(sample_dir / "layer_summary.csv", attention, video_mask)
    save_visual_scores(sample_dir / "visual_token_attention.csv", visual)
    plot_summaries(sample_dir, attention, video_mask, visual)
    overlay_contact_sheets(sample_dir, frames, visual, temporal_patch)

    final_token_id = int(input_ids[-1])
    metadata = {
        "sample_id": sample_id,
        "video_path": str(video_path),
        "utterance": row["utterance"],
        "emotion_scores": {name: float(row[name]) for name in EMOTIONS if row.get(name, "") != ""},
        "model_id": args.model_id,
        "model_revision": getattr(model.config, "_commit_hash", None),
        "analyzer_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "torch_version": torch.__version__,
        "query_position": sequence_length - 1,
        "query_token_id": final_token_id,
        "query_token": processor.tokenizer.convert_ids_to_tokens(final_token_id),
        "query_definition": "final input token after applying the chat template with add_generation_prompt=True",
        "head_reduction": "arithmetic mean over all language-decoder attention heads",
        "attention_layers": int(attention.shape[0]),
        "sequence_tokens": sequence_length,
        "visual_tokens": int(video_mask.sum()),
        "video_grid_thw_before_spatial_merge": list(grid),
        "spatial_merge_size": merge,
        "temporal_patch_size": temporal_patch,
        "decoded_frames": len(frames),
        "fps": args.fps,
        "min_frames": args.min_frames,
        "max_frames": args.max_frames,
        "frame_size": args.frame_size,
        "prompt_style": "utterance_only",
        "attention_note": (
            "Raw CSV/NPZ values are true head-mean attention probabilities. "
            "Frame-overlay colors use one linear maximum shared by every layer of this clip. "
            "Temporal and spatial summary heatmaps are normalized within each layer."
        ),
    }
    (sample_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    del inputs, attention, visual, frames, image_inputs, video_inputs
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return metadata


def main() -> None:
    args = parse_args()
    if args.fps <= 0 or args.min_frames <= 0 or args.max_frames < args.min_frames:
        raise ValueError("FPS and frame limits must be positive and ordered")
    args.index_csv = args.index_csv.expanduser().resolve()
    args.output_dir = args.output_dir.expanduser().resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = load_rows(args.index_csv, args.sample_id, args.num_clips)

    os.environ.setdefault("FORCE_QWENVL_VIDEO_READER", "decord")
    os.environ["VIDEO_MAX_PIXELS"] = str(args.frame_size**2 * args.max_frames)
    import torch
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

    if not torch.cuda.is_available():
        raise RuntimeError("This analysis requires a CUDA GPU")
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    print(f"Loading {args.model_id} with {dtype}; clips={len(rows)}", flush=True)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model_id,
        dtype=dtype,
        device_map="auto",
        attn_implementation="sdpa",
        trust_remote_code=args.trust_remote_code,
    ).eval()
    processor = AutoProcessor.from_pretrained(
        args.model_id,
        min_pixels=128 * 28 * 28,
        max_pixels=args.frame_size**2,
        trust_remote_code=args.trust_remote_code,
    )

    manifest = []
    for number, row in enumerate(rows, start=1):
        print(f"[{number}/{len(rows)}] {row['sample_id']}", flush=True)
        manifest.append(analyze_sample(args, row, model, processor))
        (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Complete: {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
