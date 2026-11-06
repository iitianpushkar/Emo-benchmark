#!/usr/bin/env python3
"""Extract several retention ratios from each MOSEI clip after one decode."""

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
from pathlib import Path

os.environ.setdefault("FORCE_QWENVL_VIDEO_READER", "decord")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cluster.extract_mosei_qwen2_5vl import log, save  # noqa: E402
from cluster.mosei_video_decoder import TimedVideoDecoder  # noqa: E402
from cluster.mosei_prompt import make_mosei_prompt  # noqa: E402
from cluster.qwen2_5_divprune import DivPruneHook  # noqa: E402
from cluster.qwen_cost import profile_forward_flops  # noqa: E402
from cluster.qwen_fastv import FastVHook  # noqa: E402
from cluster.qwen_sparsevlm import SparseVLMHook  # noqa: E402
from cluster.qwen_vispruner_attention import QwenVisionAttentionCapture, QwenVisionZipCapture  # noqa: E402
from scripts.pruning.random_prune import sample_seed  # noqa: E402


def chunk_config(args, ratio: float, protocol: dict) -> dict:
    pruning = (f"fastv_after_layer_{args.fastv_k}_video_tokens" if args.pruning == "fastv"
               else "sparsevlm_progressive_decoder_video_tokens" if args.pruning == "sparsevlm"
               else f"{args.pruning}_projected_video_tokens") if ratio < 1 else "none"
    config = {"model_dir": str(args.model_dir.resolve()), "start": args.start,
              "limit": args.limit, "video_protocol": protocol,
              "pooling": "last_token", "layer": -1, "modality": "video_text",
              "pruning": pruning, "retain_ratio": ratio,
              "profile_stride": args.profile_stride}
    if args.prompt_style == "emotion_task":
        config["prompt_style"] = "mosei_seven_class_emotion_task_v1"
    if args.pruning == "random":
        config["random_seed"] = args.random_seed
    if args.pruning == "vispruner":
        config.update(important_ratio=args.important_ratio,
                      attention_source="last_full_qwen_vision_block_mean_received_attention")
    if args.pruning == "visionzip":
        config.update(attention_source="last_full_qwen_vision_block_received_attention_and_keys",
                      attention_score_reduction="mean_heads_sum_queries_per_attention_segment_v3",
                      merging="official_qwen_65_to_5_scaled_budget_with_temporal_group_isolation_v1")
    if args.pruning == "fastv":
        config.update(fastv_k=args.fastv_k,
                      attention_source="last_prompt_query_after_k_decoder_layers")
    if args.pruning == "sparsevlm":
        config.update(sparsevlm_stages=[2, 6, 15],
                      sparsevlm_schedule="matrix_rank_lambda_geometric_with_final_budget_cap_v1",
                      attention_source="selected_post_video_text_queries_at_decoder_stages",
                      recycling="top_30_percent_discarded_then_density_cluster_mean")
    return config


def load_chunk(path: Path, config: dict, expected_ids: set[str], torch) -> dict:
    state = {"path": path, "config": config, "embeddings": [], "ids": [],
             "metadata": [], "errors": []}
    if path.is_file():
        payload = torch.load(path, map_location="cpu", weights_only=False)
        if payload.get("config") != config:
            raise ValueError(f"Existing chunk has a different configuration: {path}")
        state.update(embeddings=list(payload["embeddings"].unbind()),
                     ids=[str(value) for value in payload["sample_ids"]],
                     metadata=list(payload.get("metadata", [])),
                     errors=list(payload.get("errors", [])))
        ids = state["ids"]
        errors = state["errors"]
        error_ids = [error.get("sample_id") for error in errors]
        if (len(state["embeddings"]) != len(ids) or len(state["metadata"]) != len(ids)
                or len(set(ids)) != len(ids) or not set(ids) <= expected_ids
                or len(set(error_ids)) != len(error_ids) or not set(error_ids) <= expected_ids
                or set(ids) & set(error_ids)
                or any(not str(error.get("error", "")).startswith(
                    ("TimeoutError: Decord exceeded", "RuntimeError: Video decoding failed"))
                    for error in errors)):
            raise ValueError(f"Malformed existing chunk: {path}")
        log(f"Resuming {path}: {len(ids)} embeddings and {len(errors)} decoder failures saved")
    state["done"] = set(state["ids"]) | {error["sample_id"] for error in state["errors"]}
    state["saved_count"] = (len(state["ids"]), len(state["errors"]))
    return state


def save_chunk(state: dict) -> None:
    save(state["path"], state["embeddings"], state["ids"],
         state["metadata"], state["errors"], state["config"])
    state["saved_count"] = (len(state["ids"]), len(state["errors"]))


def extract_ratio(model, inputs, args, ratio: float, pruning_seed: int | None,
                  sample_id: str, global_row: int, common: dict, torch) -> tuple:
    torch.cuda.synchronize(model.device)
    baseline_allocated = torch.cuda.memory_allocated(model.device)
    torch.cuda.reset_peak_memory_stats(model.device)
    forward_started = time.perf_counter()
    attention = (QwenVisionAttentionCapture(model, inputs["video_grid_thw"])
                 if args.pruning == "vispruner" and ratio < 1 else
                 QwenVisionZipCapture(model, inputs["video_grid_thw"])
                 if args.pruning == "visionzip" and ratio < 1 else nullcontext())
    with torch.inference_mode(), attention as vision_attention:
        pruning_context = (FastVHook(model, inputs["input_ids"], ratio, args.fastv_k)
                           if args.pruning == "fastv" else
                           SparseVLMHook(model, inputs["input_ids"], ratio)
                           if args.pruning == "sparsevlm" else DivPruneHook(
                               model, inputs["input_ids"], ratio, method=args.pruning,
                               random_seed=pruning_seed, vision_attention=vision_attention,
                               important_ratio=args.important_ratio))
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
    if args.profile_stride and global_row % args.profile_stride == 0:
        log(f"Profiling theoretical forward FLOPs for {sample_id} at {ratio:.0%}")
        flops_profile = profile_forward_flops(
            model, inputs, inputs["input_ids"], ratio, method=args.pruning,
            random_seed=pruning_seed, important_ratio=args.important_ratio,
            fastv_k=args.fastv_k)
    metadata = {**common,
                "full_visual_tokens": pruning.full_tokens,
                "retained_visual_tokens": pruning.retained_tokens,
                "full_sequence_tokens": pruning.full_sequence_tokens,
                "retained_sequence_tokens": pruning.retained_sequence_tokens,
                "actual_retain_ratio": pruning.retained_tokens / pruning.full_tokens,
                "visionzip_dominant_tokens": pruning.visionzip_dominant_tokens if args.pruning == "visionzip" else None,
                "visionzip_contextual_tokens": pruning.visionzip_contextual_tokens if args.pruning == "visionzip" else None,
                "sparsevlm_stages": pruning.stage_metadata if args.pruning == "sparsevlm" else None,
                "random_prune_sample_seed": pruning_seed,
                "preprocess_seconds": common["preprocess_seconds"],
                "forward_seconds": forward_ended - forward_started,
                "end_to_end_seconds": common["preprocess_seconds"] + forward_ended - forward_started,
                "model_idle_allocated_bytes": baseline_allocated,
                "peak_total_allocated_bytes": peak_allocated,
                "peak_total_reserved_bytes": peak_reserved,
                "incremental_peak_allocated_bytes": max(0, peak_allocated - baseline_allocated),
                "flops_profile": flops_profile}
    return embedding, metadata


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index-csv", type=Path, required=True)
    parser.add_argument("--decoded-cache-dir", type=Path,
                        help="Use only predecoded sampled frames from this cache directory")
    parser.add_argument("--prompt-style", choices=("utterance_only", "emotion_task"),
                        default="utterance_only")
    parser.add_argument("--ratio-output", nargs=2, action="append", required=True,
                        metavar=("RATIO", "OUTPUT_PT"))
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--start", type=int, required=True)
    parser.add_argument("--limit", type=int, required=True)
    parser.add_argument("--pruning", choices=("divprune", "random", "uniform", "vispruner",
                                           "visionzip", "fastv", "sparsevlm"), required=True)
    parser.add_argument("--fastv-k", type=int, default=2)
    parser.add_argument("--important-ratio", type=float, default=0.5)
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument("--profile-stride", type=int, default=0)
    parser.add_argument("--save-every", type=int, default=25)
    parser.add_argument("--log-every", type=int, default=1)
    parser.add_argument("--decode-timeout-seconds", type=float, default=10)
    args = parser.parse_args()
    if (args.start < 0 or args.limit <= 0 or args.save_every <= 0 or args.log_every <= 0
            or args.profile_stride < 0 or args.decode_timeout_seconds <= 0
            or args.random_seed < 0 or not 0 <= args.important_ratio <= 1
            or args.fastv_k < 1):
        parser.error("Invalid extraction interval or pruning setting")
    ratio_outputs = [(float(ratio), Path(path)) for ratio, path in args.ratio_output]
    if (any(not 0 < ratio <= 1 for ratio, _ in ratio_outputs)
            or len({ratio for ratio, _ in ratio_outputs}) != len(ratio_outputs)
            or len({path for _, path in ratio_outputs}) != len(ratio_outputs)):
        parser.error("Ratios must be unique and in (0, 1], with unique output paths")
    with args.index_csv.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))[args.start:args.start + args.limit]
    if len(rows) != args.limit:
        raise ValueError(f"Expected {args.limit} index rows at offset {args.start}, found {len(rows)}")

    import torch
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

    protocol = json.loads((Path(__file__).with_name("mosei_video_protocol.json")).read_text())
    protocol["prompt_style"] = args.prompt_style
    expected_ids = {row["sample_id"] for row in rows}
    states = [load_chunk(path, chunk_config(args, ratio, protocol), expected_ids, torch)
              for ratio, path in ratio_outputs]
    log(f"Starting shared-decode extraction: rows={args.start}:{args.start + args.limit} "
        f"ratios={[ratio for ratio, _ in ratio_outputs]} pruning={args.pruning}")

    def checkpoint_on_exit() -> None:
        for state in states:
            if (len(state["ids"]), len(state["errors"])) != state["saved_count"]:
                save_chunk(state)
                log(f"Checkpoint saved: {state['path']}")

    def stop_on_sigterm(signum, _frame) -> None:
        raise SystemExit(128 + signum)

    atexit.register(checkpoint_on_exit)
    previous_sigterm = signal.signal(signal.SIGTERM, stop_on_sigterm)
    log(f"Loading Qwen2.5-VL-7B from {args.model_dir}")
    load_started = time.perf_counter()
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        str(args.model_dir), dtype=torch.bfloat16, device_map={"": 0},
        local_files_only=True, attn_implementation="sdpa").eval()
    processor = AutoProcessor.from_pretrained(str(args.model_dir), local_files_only=True)
    log(f"Model and processor ready in {time.perf_counter() - load_started:.1f}s")
    decoder = (TimedVideoDecoder(args.decode_timeout_seconds, cache_dir=args.decoded_cache_dir,
                                 cache_only=True) if args.decoded_cache_dir else
               TimedVideoDecoder(args.decode_timeout_seconds))
    atexit.register(decoder.close)
    processed = 0
    extraction_started = time.perf_counter()
    for offset, row in enumerate(rows):
        sample_id = row["sample_id"]
        pending = [(ratio, state) for (ratio, _), state in zip(ratio_outputs, states)
                   if sample_id not in state["done"]]
        if not pending:
            continue
        sample_started = time.perf_counter()
        video_path = Path(row["video_path"])
        try:
            if not video_path.is_file():
                raise FileNotFoundError(video_path)
            messages = [{"role": "user", "content": [
                {"type": "video", "video": video_path.as_uri(), "fps": protocol["fps"],
                 "min_frames": protocol["min_frames"], "max_frames": protocol["max_frames"],
                 "resized_height": protocol["frame_size"],
                 "resized_width": protocol["frame_size"]},
                {"type": "text", "text": make_mosei_prompt(row["utterance"], args.prompt_style)},
            ]}]
            prompt = processor.apply_chat_template(messages, tokenize=False,
                                                   add_generation_prompt=True)
            log(f"Decoding {sample_id} once for {len(pending)} ratios")
            images, videos, video_kwargs = (decoder.decode(messages, sample_id, video_path, protocol)
                                            if args.decoded_cache_dir else decoder.decode(messages, sample_id))
            videos, video_metadata = zip(*videos)
            frame_count = int(videos[0].shape[0])
            frame_height, frame_width = map(int, videos[0].shape[-2:])
            if (frame_height, frame_width) != (protocol["frame_size"],) * 2:
                raise ValueError(f"Unexpected frame size: {(frame_height, frame_width)}")
            inputs = processor(text=[prompt], images=images, videos=list(videos),
                               video_metadata=list(video_metadata), return_tensors="pt",
                               do_resize=False, **video_kwargs).to(model.device)
            torch.cuda.synchronize(model.device)
            preprocess_seconds = time.perf_counter() - sample_started
        except Exception as exc:
            error = {"sample_id": sample_id, "error": f"{type(exc).__name__}: {exc}"}
            for _, state in pending:
                state["errors"].append(error.copy())
                state["done"].add(sample_id)
            log(f"Decode/preprocess error for {sample_id}: {error['error']}")
        else:
            common = {"video_path": str(video_path), "utterance": row["utterance"],
                      "video_input_source": "decoded_cache" if args.decoded_cache_dir else "decord",
                      "frame_count": frame_count, "frame_height": frame_height,
                      "frame_width": frame_width, "preprocess_seconds": preprocess_seconds}
            pruning_seed = sample_seed(args.random_seed, sample_id) if args.pruning == "random" else None
            for ratio, state in pending:
                log(f"[{ratio:.0%}] Forward {sample_id}")
                try:
                    embedding, metadata = extract_ratio(
                        model, inputs, args, ratio, pruning_seed, sample_id,
                        args.start + offset, common, torch)
                except Exception as exc:
                    state["errors"].append({"sample_id": sample_id,
                                            "error": f"{type(exc).__name__}: {exc}"})
                    checkpoint_on_exit()
                    raise
                state["embeddings"].append(embedding)
                state["ids"].append(sample_id)
                state["metadata"].append(metadata)
                state["done"].add(sample_id)
            del inputs
        processed += 1
        if processed % args.log_every == 0:
            elapsed = time.perf_counter() - extraction_started
            log(f"Progress {processed} new clips; last={sample_id} elapsed={elapsed:.1f}s")
        if processed % args.save_every == 0:
            checkpoint_on_exit()
    checkpoint_on_exit()
    atexit.unregister(checkpoint_on_exit)
    atexit.unregister(decoder.close)
    decoder.close()
    signal.signal(signal.SIGTERM, previous_sigterm)
    log(f"Final shared-decode chunks saved: {[str(state['path']) for state in states]}")
    if any(state["errors"] for state in states):
        raise RuntimeError("Extraction chunks contain decoder failures; see saved chunk errors")
    if any(len(state["ids"]) != len(rows) for state in states):
        raise RuntimeError("One or more extraction chunks are incomplete")


if __name__ == "__main__":
    main()
