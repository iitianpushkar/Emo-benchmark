"""Measured extraction cost and shape-based forward-pass FLOP estimates."""

from __future__ import annotations

import math
import statistics
from pathlib import Path

FLOP_PROFILE_TARGET = 128


def profile_forward_flops(model, inputs, input_ids, retain_ratio: float,
                          method: str = "divprune", random_seed: int | None = None,
                          important_ratio: float = 0.5, fastv_k: int = 2) -> dict:
    """Count theoretical FLOPs on a separate, identically configured forward."""
    import torch
    try:
        from torch.utils.flop_counter import FlopCounterMode
    except ImportError as exc:
        raise RuntimeError("PyTorch >=2.5.1 with FlopCounterMode is required for TFLOPs reporting") from exc

    from cluster.qwen2_5_divprune import DivPruneHook
    from cluster.qwen_fastv import FastVHook
    from cluster.qwen_sparsevlm import SparseVLMHook
    from cluster.qwen_vispruner_attention import QwenVisionAttentionCapture, QwenVisionZipCapture
    from contextlib import nullcontext

    torch.cuda.synchronize(model.device)
    attention = (QwenVisionAttentionCapture(model, inputs["video_grid_thw"])
                 if method == "vispruner" and retain_ratio < 1 else
                 QwenVisionZipCapture(model, inputs["video_grid_thw"])
                 if method == "visionzip" and retain_ratio < 1 else nullcontext())
    with torch.inference_mode(), FlopCounterMode(display=False) as counter:
        with attention as vision_attention:
            pruning_context = (FastVHook(model, input_ids, retain_ratio, fastv_k)
                               if method == "fastv" else
                               SparseVLMHook(model, input_ids, retain_ratio)
                               if method == "sparsevlm" else DivPruneHook(
                                   model, input_ids, retain_ratio, method=method,
                                   random_seed=random_seed, vision_attention=vision_attention,
                                   important_ratio=important_ratio,
                               ))
            with pruning_context:
                output = model.model(**inputs, use_cache=False, return_dict=True)
                _ = output.last_hidden_state[0, -1].float().cpu()
        torch.cuda.synchronize(model.device)
    operator_flops = {str(op): int(flops) for op, flops in counter.get_flop_counts().get("Global", {}).items()}
    total_flops = int(counter.get_total_flops())
    if total_flops <= 0:
        raise RuntimeError("PyTorch FLOP counter counted no operations for this Qwen forward")
    return {
        "theoretical_forward_tflops": total_flops / 1e12,
        "operator_flops": operator_flops,
        "counted_operator_kinds": len(operator_flops),
        "scope": "vision_encoder_plus_pruning_selector_plus_language_model_forward",
        "coverage": "PyTorch FlopCounterMode shape formulas; includes registered matmul, convolution and attention (including selector work where registered), excludes unregistered/pointwise work and is not a hardware counter",
    }


def _distribution(values: list[float]) -> dict:
    ordered = sorted(values)
    return {
        "mean": statistics.mean(ordered),
        "sd": statistics.stdev(ordered) if len(ordered) > 1 else 0.0,
        "median": statistics.median(ordered),
        "p95": ordered[min(len(ordered) - 1, math.ceil(.95 * len(ordered)) - 1)],
        "max": ordered[-1],
    }


def summarize_embedding_cost(path: Path, ratio: float, expected: int) -> dict:
    import torch

    from scripts.pruning.divprune import retained_token_count

    payload = torch.load(path, map_location="cpu", weights_only=False)
    metadata = payload["metadata"]
    if len(metadata) != expected:
        raise ValueError(f"{path}: cost metadata count {len(metadata)} != {expected}")
    full = retained = full_sequence = retained_sequence = 0
    latency = {key: [] for key in ("end_to_end_seconds", "preprocess_seconds", "forward_seconds")}
    peak_allocated = []
    peak_reserved = []
    incremental_peak = []
    flop_samples = []
    profiled_ids = []
    sparsevlm_stage_outputs = None
    for sample_id, item in zip(payload["sample_ids"], metadata):
        n = int(item["full_visual_tokens"])
        k = int(item["retained_visual_tokens"])
        if k != retained_token_count(n, ratio):
            raise ValueError(f"{path}: token retention differs from requested ratio: {k}/{n}")
        full += n
        retained += k
        seq_n = int(item["full_sequence_tokens"])
        seq_k = int(item["retained_sequence_tokens"])
        if seq_k != seq_n - n + k:
            raise ValueError(f"{path}: full-sequence token count disagrees with video-token pruning")
        full_sequence += seq_n
        retained_sequence += seq_k
        stages = item.get("sparsevlm_stages")
        if stages:
            outputs = [int(stage["output_visual_tokens"]) for stage in stages]
            if len(outputs) != 3 or outputs[-1] != k:
                raise ValueError(f"{path}: invalid SparseVLM stage token counts")
            if sparsevlm_stage_outputs is None:
                sparsevlm_stage_outputs = [0, 0, 0]
            for stage_index, output_tokens in enumerate(outputs):
                sparsevlm_stage_outputs[stage_index] += output_tokens
        for key in latency:
            value = float(item[key])
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{path}: invalid {key}")
            latency[key].append(value)
        peak_allocated.append(int(item["peak_total_allocated_bytes"]))
        peak_reserved.append(int(item["peak_total_reserved_bytes"]))
        incremental_peak.append(int(item["incremental_peak_allocated_bytes"]))
        if item.get("flops_profile"):
            flop_samples.append(item["flops_profile"])
            profiled_ids.append(str(sample_id))
    if not flop_samples:
        raise ValueError(f"{path}: no FLOP profiles were recorded")
    return {
        "samples": expected,
        "full_visual_tokens": full,
        "retained_visual_tokens": retained,
        "weighted_retention": retained / full,
        "mean_full_visual_tokens_per_clip": full / expected,
        "mean_retained_visual_tokens_per_clip": retained / expected,
        "mean_full_sequence_tokens_per_clip": full_sequence / expected,
        "mean_retained_sequence_tokens_per_clip": retained_sequence / expected,
        "sparsevlm_mean_stage_output_visual_tokens": (
            [count / expected for count in sparsevlm_stage_outputs]
            if sparsevlm_stage_outputs is not None else None
        ),
        "latency_seconds_per_clip": {key: _distribution(values) for key, values in latency.items()},
        "peak_total_allocated_gib": _distribution([value / 2**30 for value in peak_allocated]),
        "peak_total_reserved_gib": _distribution([value / 2**30 for value in peak_reserved]),
        "incremental_peak_allocated_gib": _distribution([value / 2**30 for value in incremental_peak]),
        "profiled_clips": len(flop_samples),
        "profiled_sample_ids": profiled_ids,
        "mean_counted_operator_kinds": (
            statistics.mean(int(item["counted_operator_kinds"]) for item in flop_samples)
            if flop_samples else None
        ),
        "theoretical_forward_tflops_per_profiled_clip": (
            _distribution([float(item["theoretical_forward_tflops"]) for item in flop_samples])
            if flop_samples else None
        ),
        "flops_coverage": "PyTorch FlopCounterMode registered-op dense FLOPs, including attention and convolution; theoretical estimate, not measured GPU instructions",
    }
