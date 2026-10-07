#!/usr/bin/env python3
"""Deterministic random selection for projected visual-token pruning."""

from __future__ import annotations

import hashlib

import torch

from divprune import retained_token_count


DEFAULT_RETAIN_RATIO = 0.30
DEFAULT_SEED = 42
IMPLEMENTATION_NAME = "sample_id_seeded_uniform_random_projected_visual_tokens"


def sample_seed(base_seed: int, sample_id: str) -> int:
    """Derive a stable per-sample seed independent of chunking and process order."""
    digest = hashlib.blake2b(
        f"{base_seed}:{sample_id}".encode("utf-8"),
        digest_size=8,
    ).digest()
    return int.from_bytes(digest, byteorder="little", signed=False) % (2**63 - 1)


def select_random_token_indices(
    num_tokens: int,
    retain_ratio: float = DEFAULT_RETAIN_RATIO,
    seed: int = DEFAULT_SEED,
) -> torch.Tensor:
    """Uniformly sample visual-token indices, then restore sequence order."""
    target_count = retained_token_count(num_tokens, retain_ratio)
    if target_count == num_tokens:
        return torch.arange(num_tokens, dtype=torch.long)

    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    return torch.randperm(num_tokens, generator=generator)[:target_count].sort().values


def build_random_prune_sequence_indices(
    inputs_embeds: torch.Tensor,
    video_mask: torch.Tensor,
    retain_ratio: float = DEFAULT_RETAIN_RATIO,
    seed: int = DEFAULT_SEED,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return full-sequence keep indices and sampled global visual positions."""
    if inputs_embeds.ndim != 3 or inputs_embeds.shape[0] != 1:
        raise ValueError(
            "Expected inputs_embeds shaped [1, sequence_length, hidden_dim], "
            f"got {tuple(inputs_embeds.shape)}"
        )
    if video_mask.shape != inputs_embeds.shape[:2]:
        raise ValueError(
            f"video_mask shape {tuple(video_mask.shape)} does not match {tuple(inputs_embeds.shape[:2])}"
        )

    current_video_mask = video_mask.to(device=inputs_embeds.device, dtype=torch.bool)
    visual_positions = torch.nonzero(current_video_mask[0], as_tuple=False).squeeze(1)
    if visual_positions.numel() == 0:
        raise ValueError("No visual positions were found in video_mask")

    selected_local = select_random_token_indices(
        int(visual_positions.numel()),
        retain_ratio=retain_ratio,
        seed=seed,
    ).to(visual_positions.device)
    selected_visual_positions = visual_positions[selected_local]
    keep_mask = ~current_video_mask[0]
    keep_mask[selected_visual_positions] = True
    keep_indices = torch.nonzero(keep_mask, as_tuple=False).squeeze(1)
    return keep_indices, selected_visual_positions
