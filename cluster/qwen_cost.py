"""Measured extraction cost and explicitly partial profiler FLOP estimates."""

from __future__ import annotations

import math
import statistics
from pathlib import Path


def profile_supported_flops(model, inputs, input_ids, retain_ratio: float) -> dict:
    """Profile a second forward; PyTorch counts supported matmul/2D-conv ops only."""
    import torch

    from cluster.qwen2_5_divprune import DivPruneHook

    torch.cuda.synchronize()
    with torch.inference_mode(), torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
        with_flops=True,
    ) as profiler:
        with DivPruneHook(model, input_ids, retain_ratio):
            output = model.model(**inputs, use_cache=False, return_dict=True)
            _ = output.last_hidden_state[0, -1].float().cpu()
        torch.cuda.synchronize()
    events = profiler.key_averages()
    counted = [event for event in events if (event.flops or 0) > 0]
    if not counted:
        raise RuntimeError("PyTorch profiler counted no supported FLOPs for this Qwen forward")
    return {
        "supported_op_tflops": sum(event.flops for event in counted) / 1e12,
        "counted_operator_kinds": len(counted),
        "all_operator_kinds": len(events),
        "scope": "vision_encoder_plus_language_model_forward",
        "coverage": "PyTorch profiler estimates supported matrix multiplication and 2D convolution only; not total model FLOPs",
    }


def _distribution(values: list[float]) -> dict:
    ordered = sorted(values)
    return {
        "mean": statistics.mean(ordered),
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
    for item in metadata:
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
        "latency_seconds_per_clip": {key: _distribution(values) for key, values in latency.items()},
        "peak_total_allocated_gib": _distribution([value / 2**30 for value in peak_allocated]),
        "peak_total_reserved_gib": _distribution([value / 2**30 for value in peak_reserved]),
        "incremental_peak_allocated_gib": _distribution([value / 2**30 for value in incremental_peak]),
        "profiled_clips": len(flop_samples),
        "mean_counted_operator_kinds": (
            statistics.mean(int(item["counted_operator_kinds"]) for item in flop_samples)
            if flop_samples else None
        ),
        "mean_all_operator_kinds": (
            statistics.mean(int(item["all_operator_kinds"]) for item in flop_samples)
            if flop_samples else None
        ),
        "supported_op_tflops_per_profiled_clip": (
            _distribution([float(item["supported_op_tflops"]) for item in flop_samples])
            if flop_samples else None
        ),
        "flops_coverage": "PyTorch profiler matmul/2D-conv estimate; excludes unsupported operators and is not full-model TFLOPs",
    }
