#!/usr/bin/env python3
"""Make per-layer, every-frame full-baseline attention overlays for one clip."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import numpy as np  # noqa: E402

from cluster.meld_prompt import make_meld_prompt  # noqa: E402
from cluster.mosei_prompt import make_mosei_prompt  # noqa: E402
from cluster.mosei_cache_subset import resolve_cache_frames  # noqa: E402
from cluster.mosei_decoded_cache import cache_available, identity  # noqa: E402
from cluster.mosei_video_decoder import TimedVideoDecoder  # noqa: E402
from cluster.preflight import verify_model  # noqa: E402
from cluster.verify_qwen_environment import require_cuda_gpu, verify_qwen_environment  # noqa: E402
from scripts.analysis.qwen_decoder_attention import QwenLastQueryAttentionCapture  # noqa: E402
from scripts.analysis.visualize_mosei_decoder_attention import (  # noqa: E402
    plot_summaries, rgb_frames, safe_name, save_layer_summary, save_visual_scores,
)


def find_sample(index_csv: Path, sample_id: str) -> dict[str, str]:
    with index_csv.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if not {"sample_id", "video_path", "utterance"} <= set(reader.fieldnames or ()):
            raise ValueError(f"{index_csv}: missing sample_id, video_path, or utterance")
        matches = [row for row in reader if row["sample_id"] == sample_id]
    if len(matches) != 1:
        raise ValueError(f"{index_csv}: expected one row for {sample_id}, found {len(matches)}")
    return matches[0]


def frame_group_indices(frame_count: int, temporal_groups: int,
                        temporal_patch_size: int) -> list[int]:
    if frame_count != temporal_groups * temporal_patch_size:
        raise ValueError("Decoded frames do not match Qwen's temporal video grid")
    return [frame_index // temporal_patch_size for frame_index in range(frame_count)]


def save_every_frame_overlays(output_dir: Path, frames: np.ndarray, visual: np.ndarray,
                              temporal_patch_size: int) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    groups = frame_group_indices(len(frames), visual.shape[1], temporal_patch_size)
    overlay_dir = output_dir / "layer_overlays"
    overlay_dir.mkdir(parents=True, exist_ok=True)
    columns = min(8, len(frames))
    rows = math.ceil(len(frames) / columns)
    shared_max = max(float(visual.max()), 1e-30)
    for layer in range(visual.shape[0]):
        figure, axes = plt.subplots(rows, columns, figsize=(2.45 * columns, 2.65 * rows),
                                    squeeze=False)
        for frame_index, axis in enumerate(axes.flat):
            if frame_index >= len(frames):
                axis.axis("off")
                continue
            group = groups[frame_index]
            heat = visual[layer, group] / shared_max
            axis.imshow(frames[frame_index])
            axis.imshow(heat, cmap="inferno", vmin=0, vmax=1,
                        interpolation="bilinear", alpha=np.clip(heat * 0.7, 0, 0.7))
            axis.set_title(f"Frame {frame_index} · group {group}", fontsize=9)
            axis.set_xticks([])
            axis.set_yticks([])
        figure.suptitle(f"Decoder layer {layer:02d} · full visual tokens · shared color scale",
                        fontsize=12)
        figure.tight_layout(rect=(0, 0, 1, 0.97))
        figure.savefig(overlay_dir / f"layer_{layer:02d}.png", dpi=150)
        plt.close(figure)
        print(f"Saved attention overlay for decoder layer {layer:02d}", flush=True)


def main(dataset: str = "meld") -> int:
    if dataset not in ("meld", "mosei"):
        raise ValueError(f"Unknown dataset: {dataset}")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=REPO.parents[1])
    parser.add_argument("--sample-id", required=True,
                        help=f"Exact {dataset.upper()} sample_id from an index")
    splits = ("train", "dev", "test") if dataset == "meld" else ("train", "valid", "test")
    parser.add_argument("--split", choices=splits, default="test")
    if dataset == "mosei":
        parser.add_argument("--prompt-style", choices=("emotion_task", "emotion_multilabel_task"),
                            default="emotion_multilabel_task",
                            help="Match the seven-class or multilabel full baseline")
    parser.add_argument("--index-csv", type=Path,
                        help="Default: baseline index, or predecode index if baseline is not prepared")
    parser.add_argument("--baseline-dir", type=Path)
    parser.add_argument("--model-dir", type=Path)
    parser.add_argument("--decoded-cache-dir", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()

    root = args.root.expanduser().resolve()
    prompt_style = "meld_emotion_task" if dataset == "meld" else args.prompt_style
    default_baseline = ("qwen2_5_vl_7b_meld_emotion_prompt_cache_success" if dataset == "meld"
                        else "qwen2_5_vl_7b_mosei_single_label_emotion_prompt_cache_success"
                        if prompt_style == "emotion_task"
                        else "qwen2_5_vl_7b_mosei_multilabel_emotion_prompt_cache_success")
    baseline_dir = (args.baseline_dir or root / "results" / default_baseline).expanduser().resolve()
    cache_root = (args.decoded_cache_dir or root / "datasets" / dataset /
                  "decoded_qwen_6fps_224_v1").expanduser().resolve()
    cache_dir = resolve_cache_frames(cache_root)
    baseline_index = baseline_dir / "indexes" / f"{dataset}_{args.split}_index.csv"
    predecode_index = cache_dir.parent / "indexes" / f"{dataset}_{args.split}_index.csv"
    index_csv = (args.index_csv or (baseline_index if baseline_index.is_file()
                                    else predecode_index)).expanduser().resolve()
    model_dir = (args.model_dir or root / "models" / "qwen").expanduser().resolve()
    output_dir = (args.output_dir or baseline_dir / "attention_heatmaps" /
                  safe_name(args.sample_id)).expanduser().resolve()
    row = find_sample(index_csv, args.sample_id)
    video_path = Path(row["video_path"]).expanduser().resolve(strict=True)
    protocol = json.loads(Path(__file__).with_name(f"{dataset}_video_protocol.json").read_text())
    protocol["prompt_style"] = prompt_style
    if not cache_available(cache_dir, identity(args.sample_id, video_path, protocol)):
        raise ValueError(f"No valid decoded-frame cache entry for {args.sample_id}")
    model_info = verify_model(model_dir, "qwen")
    if args.verify_only:
        print(json.dumps({"dataset": dataset, "sample_id": args.sample_id, "split": args.split,
                          "video_path": str(video_path), "index_csv": str(index_csv),
                          "cache_dir": str(cache_dir), "model": model_info,
                          "prompt_style": prompt_style,
                          "output_dir": str(output_dir)}, indent=2))
        return 0

    os.environ["HF_HUB_OFFLINE"] = "1"
    environment = verify_qwen_environment(curve=True)
    import torch
    import transformers
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

    if transformers.__version__ != "4.57.1":
        raise RuntimeError("This visualization requires transformers==4.57.1")
    require_cuda_gpu(torch)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        str(model_dir), dtype=torch.bfloat16, device_map={"": 0},
        local_files_only=True, attn_implementation="sdpa").eval()
    processor = AutoProcessor.from_pretrained(str(model_dir), local_files_only=True)
    messages = [{"role": "user", "content": [
        {"type": "video", "video": video_path.as_uri(),
         "fps": protocol["fps"], "min_frames": protocol["min_frames"],
         "max_frames": protocol["max_frames"],
         "resized_height": protocol["frame_size"],
         "resized_width": protocol["frame_size"]},
        {"type": "text", "text": (make_meld_prompt(row["utterance"])
                                  if dataset == "meld" else
                                  make_mosei_prompt(row["utterance"], prompt_style))},
    ]}]
    prompt = processor.apply_chat_template(messages, tokenize=False,
                                           add_generation_prompt=True)
    decoder = TimedVideoDecoder(10, cache_dir=cache_dir, cache_only=True)
    try:
        images, videos, video_kwargs = decoder.decode(messages, args.sample_id,
                                                       video_path, protocol)
    finally:
        decoder.close()
    frames = rgb_frames(videos[0])
    video_tensors, video_metadata = zip(*videos)
    inputs = processor(text=[prompt], images=images, videos=list(video_tensors),
                       video_metadata=list(video_metadata), return_tensors="pt",
                       do_resize=False, **video_kwargs).to(model.device)
    sequence_length = int(inputs["input_ids"].shape[-1])
    with torch.inference_mode(), QwenLastQueryAttentionCapture(model, sequence_length) as capture:
        model.model(**inputs, use_cache=False, return_dict=True)
    attention = capture.stacked().numpy()
    input_ids = inputs["input_ids"][0].detach().cpu().numpy()
    video_mask = input_ids == model.config.video_token_id
    if not video_mask.any():
        raise ValueError("No visual tokens in the full baseline input")
    temporal, height, width = map(int, inputs["video_grid_thw"][0].detach().cpu().tolist())
    merge = int(model.config.vision_config.spatial_merge_size)
    temporal_patch = int(model.config.vision_config.temporal_patch_size)
    if height % merge or width % merge:
        raise ValueError("Video grid is incompatible with spatial merging")
    grid_rows, grid_columns = height // merge, width // merge
    if int(video_mask.sum()) != temporal * grid_rows * grid_columns:
        raise ValueError("Visual token count does not match Qwen video grid")
    frame_group_indices(len(frames), temporal, temporal_patch)
    visual = attention[:, video_mask].reshape(attention.shape[0], temporal,
                                               grid_rows, grid_columns)

    output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_dir / "attention_arrays.npz", mean_attention=attention,
                        video_attention=visual, input_ids=input_ids, video_mask=video_mask,
                        video_grid_thw=np.asarray((temporal, height, width)))
    save_layer_summary(output_dir / "layer_summary.csv", attention, video_mask)
    save_visual_scores(output_dir / "visual_token_attention.csv", visual)
    plot_summaries(output_dir, attention, video_mask, visual)
    save_every_frame_overlays(output_dir, frames, visual, temporal_patch)
    (output_dir / "metadata.json").write_text(json.dumps({
        "dataset": dataset, "sample_id": args.sample_id, "split": args.split,
        "video_path": str(video_path),
        "index_csv": str(index_csv), "cache_dir": str(cache_dir),
        "model_dir": str(model_dir), "prompt_style": prompt_style,
        "query": "final prompt token after Qwen chat template",
        "head_reduction": "mean over attention heads",
        "layers": int(attention.shape[0]), "frames": len(frames),
        "video_grid_thw": [temporal, height, width],
        "spatial_merge_size": merge, "temporal_patch_size": temporal_patch,
        "overlay_scale": "one raw attention maximum shared across all layers and frames",
        "environment": environment,
    }, indent=2) + "\n")
    print(f"Saved {attention.shape[0]} per-layer overlays of all {len(frames)} frames to "
          f"{output_dir / 'layer_overlays'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
