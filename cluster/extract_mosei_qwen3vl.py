#!/usr/bin/env python3
"""Extract frozen Qwen3-VL video and transcript embeddings for one MOSEI chunk."""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path

os.environ.setdefault("FORCE_QWENVL_VIDEO_READER", "decord")


def save(path, embeddings, ids, metadata, errors, config):
    import torch

    payload = {
        "embeddings": torch.stack(embeddings) if embeddings else torch.empty((0, 4096)),
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
    parser.add_argument("--save-every", type=int, default=25)
    args = parser.parse_args()
    if args.start < 0 or args.limit <= 0 or args.save_every <= 0:
        parser.error("--start must be nonnegative and --limit/--save-every must be positive")

    with args.index_csv.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))[args.start:args.start + args.limit]
    if len(rows) != args.limit:
        raise ValueError(f"Expected {args.limit} index rows at offset {args.start}, found {len(rows)}")

    import torch
    from qwen_vl_utils import process_vision_info
    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

    video_protocol = json.loads((Path(__file__).with_name("mosei_video_protocol.json")).read_text())
    config = {"model_dir": str(args.model_dir.resolve()), "start": args.start,
              "limit": args.limit, "video_protocol": video_protocol,
              "pooling": "last_token", "layer": -1, "modality": "video_text"}
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
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        str(args.model_dir), dtype=torch.bfloat16, device_map="auto", local_files_only=True,
        attn_implementation="sdpa",
    ).eval()
    processor = AutoProcessor.from_pretrained(str(args.model_dir), local_files_only=True)
    processed = 0
    for row in rows:
        sample_id = row["sample_id"]
        if sample_id in done:
            continue
        try:
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
                messages, image_patch_size=16, return_video_kwargs=True,
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
            with torch.inference_mode():
                output = model.model(**inputs, use_cache=False, return_dict=True)
                embedding = output.last_hidden_state[0, -1].float().cpu()
            if embedding.shape != (4096,) or not torch.isfinite(embedding).all():
                raise ValueError(f"Invalid embedding shape or values: {tuple(embedding.shape)}")
            embeddings.append(embedding)
            ids.append(sample_id)
            metadata.append({"video_path": str(video_path), "utterance": row["utterance"],
                             "frame_count": frame_count,
                             "frame_height": frame_height, "frame_width": frame_width})
            done.add(sample_id)
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
