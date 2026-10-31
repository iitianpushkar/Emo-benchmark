#!/usr/bin/env python3
"""Extract frozen Qwen2.5-VL video/transcript embeddings and per-clip cost."""

from __future__ import annotations

import argparse
import atexit
import csv
import json
import os
import signal
import sys
import time
from contextlib import nullcontext
from datetime import datetime, timezone
from pathlib import Path

os.environ.setdefault("FORCE_QWENVL_VIDEO_READER", "decord")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cluster.qwen2_5_divprune import DivPruneHook  # noqa: E402
from cluster.qwen_cost import profile_forward_flops  # noqa: E402
from cluster.qwen_fastv import FastVHook  # noqa: E402
from cluster.qwen_sparsevlm import SparseVLMHook  # noqa: E402
from cluster.qwen_vispruner_attention import QwenVisionAttentionCapture, QwenVisionZipCapture  # noqa: E402
from scripts.pruning.random_prune import sample_seed  # noqa: E402


def log(message: str) -> None:
    print(f"[{datetime.now(timezone.utc).isoformat()}] {message}", flush=True)


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
                        help="Fraction of projected video tokens retained; 1 is unpruned")
    parser.add_argument("--pruning", choices=("divprune", "random", "uniform", "vispruner", "visionzip", "fastv", "sparsevlm"), default="divprune")
    parser.add_argument("--fastv-k", type=int, default=2,
                        help="FastV decoder layers processed before pruning (default: 2)")
    parser.add_argument("--important-ratio", type=float, default=0.5,
                        help="VisPruner fraction of kept tokens selected by vision attention")
    parser.add_argument("--random-seed", type=int, default=42,
                        help="Base seed for sample-ID-stable random token selection")
    parser.add_argument("--profile-stride", type=int, default=0,
                        help="Profile FLOPs for global index 0 and each stride thereafter; 0 disables profiling")
    parser.add_argument("--save-every", type=int, default=25)
    parser.add_argument("--log-every", type=int, default=1,
                        help="Print clip progress every N new attempts (default: 1)")
    args = parser.parse_args()
    if (args.start < 0 or args.limit <= 0 or args.save_every <= 0
            or args.log_every <= 0 or args.profile_stride < 0):
        parser.error("--start must be nonnegative and --limit/--save-every/--log-every must be positive")
    if not 0 < args.retain_ratio <= 1:
        parser.error("--retain-ratio must be in (0, 1]")
    if args.random_seed < 0:
        parser.error("--random-seed must be nonnegative")
    if not 0 <= args.important_ratio <= 1:
        parser.error("--important-ratio must be in [0, 1]")
    if args.fastv_k < 1:
        parser.error("--fastv-k must be positive")

    with args.index_csv.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))[args.start:args.start + args.limit]
    if len(rows) != args.limit:
        raise ValueError(f"Expected {args.limit} index rows at offset {args.start}, found {len(rows)}")

    log(f"Starting extraction: index={args.index_csv} rows={args.start}:{args.start + args.limit} "
        f"retention={args.retain_ratio:.0%} pruning={args.pruning} "
        f"random_seed={args.random_seed if args.pruning == 'random' else 'n/a'} "
        f"important_ratio={args.important_ratio if args.pruning == 'vispruner' else 'n/a'} "
        f"fastv_k={args.fastv_k if args.pruning == 'fastv' else 'n/a'} "
        f"output={args.output_pt}")
    import torch
    from qwen_vl_utils import process_vision_info
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

    video_protocol = json.loads((Path(__file__).with_name("mosei_video_protocol.json")).read_text())
    config = {"model_dir": str(args.model_dir.resolve()), "start": args.start,
              "limit": args.limit, "video_protocol": video_protocol,
              "pooling": "last_token", "layer": -1, "modality": "video_text",
              "pruning": ((f"fastv_after_layer_{args.fastv_k}_video_tokens"
                           if args.pruning == "fastv" else
                           "sparsevlm_progressive_decoder_video_tokens" if args.pruning == "sparsevlm"
                           else f"{args.pruning}_projected_video_tokens")
                          if args.retain_ratio < 1 else "none"),
              "retain_ratio": args.retain_ratio, "profile_stride": args.profile_stride}
    if args.pruning == "random":
        config["random_seed"] = args.random_seed
    if args.pruning == "vispruner":
        config["important_ratio"] = args.important_ratio
        config["attention_source"] = "last_full_qwen_vision_block_mean_received_attention"
    if args.pruning == "visionzip":
        config["attention_source"] = "last_full_qwen_vision_block_received_attention_and_keys"
        config["merging"] = "official_qwen_65_to_5_scaled_budget_with_temporal_group_isolation_v1"
    if args.pruning == "fastv":
        config["fastv_k"] = args.fastv_k
        config["attention_source"] = "last_prompt_query_after_k_decoder_layers"
    if args.pruning == "sparsevlm":
        config["sparsevlm_stages"] = [2, 6, 15]
        config["sparsevlm_schedule"] = "matrix_rank_lambda_geometric_with_final_budget_cap_v1"
        config["attention_source"] = "selected_post_video_text_queries_at_decoder_stages"
        config["recycling"] = "top_30_percent_discarded_then_density_cluster_mean"
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
        log(f"Resuming {args.output_pt}: {len(ids)} embeddings already saved")

    done = set(ids)
    saved_state = [len(ids), len(errors)]

    def checkpoint_on_exit() -> None:
        complete = min(len(embeddings), len(ids), len(metadata))
        if (complete, len(errors)) != tuple(saved_state):
            save(args.output_pt, embeddings[:complete], ids[:complete],
                 metadata[:complete], errors, config)
            log(f"Interrupted: saved {complete}/{len(rows)} embeddings to {args.output_pt}")

    def stop_on_sigterm(signum, _frame) -> None:
        raise SystemExit(128 + signum)

    atexit.register(checkpoint_on_exit)
    previous_sigterm = signal.signal(signal.SIGTERM, stop_on_sigterm)
    log(f"Loading Qwen2.5-VL-7B from {args.model_dir}")
    load_started = time.perf_counter()
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        str(args.model_dir), dtype=torch.bfloat16, device_map={"": 0}, local_files_only=True,
        attn_implementation="sdpa",
    ).eval()
    processor = AutoProcessor.from_pretrained(str(args.model_dir), local_files_only=True)
    log(f"Model and processor ready in {time.perf_counter() - load_started:.1f}s; "
        f"GPU={torch.cuda.get_device_name(model.device)}")
    processed = 0
    extraction_started = time.perf_counter()
    for offset, row in enumerate(rows):
        sample_id = row["sample_id"]
        if sample_id in done:
            continue
        pruning_seed = sample_seed(args.random_seed, sample_id) if args.pruning == "random" else None
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
            attention = (QwenVisionAttentionCapture(model, inputs["video_grid_thw"])
                         if args.pruning == "vispruner" and args.retain_ratio < 1 else
                         QwenVisionZipCapture(model, inputs["video_grid_thw"])
                         if args.pruning == "visionzip" and args.retain_ratio < 1 else nullcontext())
            with torch.inference_mode(), attention as vision_attention:
                pruning_context = (FastVHook(model, inputs["input_ids"], args.retain_ratio, args.fastv_k)
                                   if args.pruning == "fastv" else
                                   SparseVLMHook(model, inputs["input_ids"], args.retain_ratio)
                                   if args.pruning == "sparsevlm" else DivPruneHook(
                                       model, inputs["input_ids"], args.retain_ratio,
                                       method=args.pruning, random_seed=pruning_seed,
                                       vision_attention=vision_attention, important_ratio=args.important_ratio,
                                   ))
                with pruning_context as pruning:
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
                log(f"Profiling theoretical forward FLOPs for {sample_id} "
                    f"(global row {args.start + offset})")
                flops_profile = profile_forward_flops(
                    model, inputs, inputs["input_ids"], args.retain_ratio,
                    method=args.pruning, random_seed=pruning_seed,
                    important_ratio=args.important_ratio,
                    fastv_k=args.fastv_k,
                )
                log(f"FLOP profile complete for {sample_id}")
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
                             "visionzip_dominant_tokens": pruning.visionzip_dominant_tokens if args.pruning == "visionzip" else None,
                             "visionzip_contextual_tokens": pruning.visionzip_contextual_tokens if args.pruning == "visionzip" else None,
                             "sparsevlm_stages": pruning.stage_metadata if args.pruning == "sparsevlm" else None,
                             "random_prune_sample_seed": pruning_seed,
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
            log(f"Extraction error for {sample_id}: {type(exc).__name__}: {exc}")
        processed += 1
        if processed % args.log_every == 0 or len(done) == len(rows):
            elapsed = time.perf_counter() - extraction_started
            rate = processed / elapsed if elapsed else 0.0
            remaining = max(0, len(rows) - len(done) - len(errors))
            eta = remaining / rate if rate else float("inf")
            log(f"Progress {len(done)}/{len(rows)} embeddings; "
                f"new_attempts={processed} errors={len(errors)} last={sample_id} "
                f"elapsed={elapsed:.1f}s rate={rate:.3f} clips/s ETA={eta:.0f}s")
        if processed % args.save_every == 0:
            save(args.output_pt, embeddings, ids, metadata, errors, config)
            saved_state[:] = [len(ids), len(errors)]
            log(f"Checkpoint saved: {len(ids)}/{len(rows)} embeddings in {args.output_pt}")
    save(args.output_pt, embeddings, ids, metadata, errors, config)
    saved_state[:] = [len(ids), len(errors)]
    atexit.unregister(checkpoint_on_exit)
    signal.signal(signal.SIGTERM, previous_sigterm)
    log(f"Final chunk saved: {len(ids)}/{len(rows)} embeddings; errors={len(errors)}")
    if errors:
        raise RuntimeError(f"{len(errors)} extraction errors in {args.output_pt}; first: {errors[:3]}")
    if len(ids) != len(rows):
        raise RuntimeError(f"Incomplete chunk {args.output_pt}: {len(ids)}/{len(rows)}")


if __name__ == "__main__":
    main()
