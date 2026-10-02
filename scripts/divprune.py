#!/usr/bin/env python3
"""Paper-faithful max-min diversity selection for visual token pruning.

Reference: https://github.com/vbdi/divprune at REFERENCE_COMMIT.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


DEFAULT_RETAIN_RATIO = 0.098
IMPLEMENTATION_NAME = "official_layer0_projected_visual_max_min_cosine"
REFERENCE_COMMIT = "799e2d950aa01ba7860907f5a6d86061f885dca6"


def retained_token_count(num_tokens: int, retain_ratio: float) -> int:
    """Return the bounded number of visual tokens retained by DivPrune."""
    if num_tokens <= 0:
        raise ValueError("num_tokens must be positive")
    if not 0.0 < retain_ratio <= 1.0:
        raise ValueError("retain_ratio must be in the interval (0, 1]")
    return min(num_tokens, max(1, int(round(num_tokens * retain_ratio))))


def select_diverse_token_indices(
    visual_tokens: torch.Tensor,
    retain_ratio: float = DEFAULT_RETAIN_RATIO,
) -> torch.Tensor:
    """Select a deterministic max-min-diverse subset using cosine distance.

    The returned indices are sorted into their original sequence order. As in
    the authors' implementation, selection starts from the token with the
    largest second-smallest distance (its nearest non-self neighbor), then adds
    the token whose minimum distance to the selected set is largest.
    """
    if visual_tokens.ndim != 2:
        raise ValueError(
            "Expected visual_tokens shaped [num_tokens, hidden_dim], "
            f"got {tuple(visual_tokens.shape)}"
        )
    if visual_tokens.shape[0] == 0 or visual_tokens.shape[1] == 0:
        raise ValueError("visual_tokens must have non-zero token and feature dimensions")
    if not torch.isfinite(visual_tokens).all():
        raise ValueError("visual_tokens contain NaN or Inf values")

    num_tokens = visual_tokens.shape[0]
    target_count = retained_token_count(num_tokens, retain_ratio)
    device = visual_tokens.device
    if target_count == num_tokens:
        return torch.arange(num_tokens, device=device, dtype=torch.long)

    normalized = F.normalize(visual_tokens.float(), p=2, dim=-1, eps=1e-12)
    distances = (1.0 - normalized @ normalized.transpose(0, 1)).clamp_(0.0, 2.0)
    nearest_non_self_distance = torch.topk(
        distances,
        k=min(2, num_tokens),
        dim=0,
        largest=False,
    ).values[-1]
    first = int(nearest_non_self_distance.argmax().item())
    selected = torch.zeros(num_tokens, dtype=torch.bool, device=device)
    selected[first] = True
    min_distance = distances[first]
    min_distance[selected] = float("-inf")

    for _ in range(1, target_count):
        next_index = int(min_distance.argmax().item())
        selected[next_index] = True
        min_distance = torch.minimum(min_distance, distances[next_index])
        min_distance[selected] = float("-inf")

    return torch.nonzero(selected, as_tuple=False).squeeze(1).sort().values.long()


def build_divprune_sequence_indices(
    inputs_embeds: torch.Tensor,
    video_mask: torch.Tensor,
    retain_ratio: float = DEFAULT_RETAIN_RATIO,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return full-sequence keep indices and selected global visual positions."""
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
    selected_local = select_diverse_token_indices(
        inputs_embeds[0, visual_positions],
        retain_ratio=retain_ratio,
    )
    selected_visual_positions = visual_positions[selected_local]
    keep_mask = ~current_video_mask[0]
    keep_mask[selected_visual_positions] = True
    keep_indices = torch.nonzero(keep_mask, as_tuple=False).squeeze(1)
    return keep_indices, selected_visual_positions
