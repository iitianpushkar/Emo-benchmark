#!/usr/bin/env python3
"""Extract frozen Qwen2.5-VL video/transcript embeddings and per-clip cost."""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("FORCE_QWENVL_VIDEO_READER", "decord")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cluster.qwen2_5_divprune import DivPruneHook  # noqa: E402
from cluster.qwen_cost import profile_forward_flops  # noqa: E402


def save(path, embeddings, ids, metadata, errors, config):
    import torch

    payload = {
        "embeddings": torch.stack(embeddings) if embeddings else torch.empty((0, 3584)),
        "labels": torch.zeros(len(ids), dtype=torch.long),
        "sample_ids": ids,
        "metadata": metadata,
        "errors": errors,
        "config": config,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".pt.tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index-csv", type=Path, required=True)
    parser.add_argument("--output-pt", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--start", type=int, required=True)
    parser.add_argument("--limit", type=int, required=True)
    parser.add_argument("--retain-ratio", type=float, default=1.0,
                        help="Fraction of projected video tokens retained by DivPrune; 1 is unpruned")
    parser.add_argument("--profile-stride", type=int, default=0,
                        help="Profile FLOPs for global index 0 and each stride thereafter; 0 disables profiling")
    parser.add_argument("--save-every", type=int, default=25)
    args = parser.parse_args()
    if args.start < 0 or args.limit <= 0 or args.save_every <= 0 or args.profile_stride < 0:
        parser.error("--start must be nonnegative and --limit/--save-every must be positive")
    if not 0 < args.retain_ratio <= 1:
        parser.error("--retain-ratio must be in (0, 1]")

    with args.index_csv.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))[args.start:args.start + args.limit]
    if len(rows) != args.limit:
        raise ValueError(f"Expected {args.limit} index rows at offset {args.start}, found {len(rows)}")

    import torch
    from qwen_vl_utils import process_vision_info
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

    video_protocol = json.loads((Path(__file__).with_name("mosei_video_protocol.json")).read_text())
    config = {"model_dir": str(args.model_dir.resolve()), "start": args.start,
              "limit": args.limit, "video_protocol": video_protocol,
              "pooling": "last_token", "layer": -1, "modality": "video_text",
              "pruning": "divprune_projected_video_tokens" if args.retain_ratio < 1 else "none",
              "retain_ratio": args.retain_ratio, "profile_stride": args.profile_stride}
    embeddings, ids, metadata, errors = [], [], [], []
    if args.output_pt.is_file():
        payload = torch.load(args.output_pt, map_location="cpu", weights_only=False)
        if payload.get("config") != config:
            raise ValueError(f"Existing chunk has a different configuration: {args.output_pt}")
        embeddings = list(payload["embeddings"].unbind())
        ids = [str(value) for value in payload["sample_ids"]]
        metadata = list(payload.get("metadata", []))
        if len(embeddings) != len(ids) or len(metadata) != len(ids):
            raise ValueError(f"Malformed existing chunk: {args.output_pt}")
        expected_ids = {row["sample_id"] for row in rows}
        if len(set(ids)) != len(ids) or not set(ids) <= expected_ids:
            raise ValueError(f"Existing chunk has unexpected or repeated IDs: {args.output_pt}")
        print(f"Resuming {args.output_pt}: {len(ids)} embeddings", flush=True)

    done = set(ids)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        str(args.model_dir), dtype=torch.bfloat16, device_map={"": 0}, local_files_only=True,
        attn_implementation="sdpa",
    ).eval()
    processor = AutoProcessor.from_pretrained(str(args.model_dir), local_files_only=True)
    processed = 0
    for offset, row in enumerate(rows):
        sample_id = row["sample_id"]
        if sample_id in done:
            continue
        try:
            sample_started = time.perf_counter()
            video_path = Path(row["video_path"])
            if not video_path.is_file():
                raise FileNotFoundError(video_path)
            messages = [{"role": "user", "content": [
                {"type": "video", "video": video_path.as_uri(), "fps": video_protocol["fps"],
                 "min_frames": video_protocol["min_frames"], "max_frames": video_protocol["max_frames"],
                 "resized_height": video_protocol["frame_size"],
                 "resized_width": video_protocol["frame_size"]},
                {"type": "text", "text": row["utterance"]},
            ]}]
            prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            images, videos, video_kwargs = process_vision_info(
                messages, image_patch_size=14, return_video_kwargs=True,
                return_video_metadata=True,
            )
            videos, video_metadata = zip(*videos)
            frame_count = int(videos[0].shape[0])
            frame_height, frame_width = map(int, videos[0].shape[-2:])
            if (frame_height, frame_width) != (video_protocol["frame_size"],) * 2:
                raise ValueError(
                    f"Actual frame size {(frame_height, frame_width)} differs from "
                    f"shared protocol {video_protocol['frame_size']} square"
                )
            inputs = processor(
                text=[prompt], images=images, videos=list(videos),
                video_metadata=list(video_metadata), return_tensors="pt",
                do_resize=False, **video_kwargs,
            ).to(model.device)
            torch.cuda.synchronize(model.device)
            preprocess_ended = time.perf_counter()
            baseline_allocated = torch.cuda.memory_allocated(model.device)
            torch.cuda.reset_peak_memory_stats(model.device)
            forward_started = time.perf_counter()
            with torch.inference_mode(), DivPruneHook(model, inputs["input_ids"], args.retain_ratio) as pruning:
                output = model.model(**inputs, use_cache=False, return_dict=True)
                embedding = output.last_hidden_state[0, -1].float().cpu()
            torch.cuda.synchronize(model.device)
            forward_ended = time.perf_counter()
            peak_allocated = torch.cuda.max_memory_allocated(model.device)
            peak_reserved = torch.cuda.max_memory_reserved(model.device)
            del output
            if embedding.shape != (3584,) or not torch.isfinite(embedding).all():
                raise ValueError(f"Invalid embedding shape or values: {tuple(embedding.shape)}")
            flops_profile = None
            if args.profile_stride and (args.start + offset) % args.profile_stride == 0:
                flops_profile = profile_forward_flops(model, inputs, inputs["input_ids"], args.retain_ratio)
            embeddings.append(embedding)
            ids.append(sample_id)
            metadata.append({"video_path": str(video_path), "utterance": row["utterance"],
                             "frame_count": frame_count,
                             "frame_height": frame_height, "frame_width": frame_width,
                             "full_visual_tokens": pruning.full_tokens,
                             "retained_visual_tokens": pruning.retained_tokens,
                             "full_sequence_tokens": pruning.full_sequence_tokens,
                             "retained_sequence_tokens": pruning.retained_sequence_tokens,
                             "actual_retain_ratio": pruning.retained_tokens / pruning.full_tokens,
                             "preprocess_seconds": preprocess_ended - sample_started,
                             "forward_seconds": forward_ended - forward_started,
                             "end_to_end_seconds": forward_ended - sample_started,
                             "model_idle_allocated_bytes": baseline_allocated,
                             "peak_total_allocated_bytes": peak_allocated,
                             "peak_total_reserved_bytes": peak_reserved,
                             "incremental_peak_allocated_bytes": max(0, peak_allocated - baseline_allocated),
                             "flops_profile": flops_profile})
            done.add(sample_id)
            del inputs, pruning
        except Exception as exc:
            errors.append({"sample_id": sample_id, "error": f"{type(exc).__name__}: {exc}"})
            print(f"Extraction error for {sample_id}: {exc}", flush=True)
        processed += 1
        if processed % args.save_every == 0:
            save(args.output_pt, embeddings, ids, metadata, errors, config)
            print(f"Saved {len(ids)}/{len(rows)} embeddings in {args.output_pt}", flush=True)
    save(args.output_pt, embeddings, ids, metadata, errors, config)
    if errors:
        raise RuntimeError(f"{len(errors)} extraction errors in {args.output_pt}; first: {errors[:3]}")
    if len(ids) != len(rows):
        raise RuntimeError(f"Incomplete chunk {args.output_pt}: {len(ids)}/{len(rows)}")


if __name__ == "__main__":
    main()
