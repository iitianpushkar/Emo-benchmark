"""Deterministic temporal-stratified, spatially uniform visual-token selection."""

from __future__ import annotations

import torch

from scripts.pruning.divprune import retained_token_count


def _even_indices(count: int, keep: int, device: torch.device) -> torch.Tensor:
    """Select bin centers without duplicates, in ascending order."""
    if not 0 < keep <= count:
        raise ValueError(f"Invalid systematic selection: keep={keep}, count={count}")
    ranks = torch.arange(keep, device=device, dtype=torch.long)
    return ((2 * ranks + 1) * count) // (2 * keep)


def _temporal_quotas(counts: list[int], keep: int) -> list[int]:
    """Give every temporal group one token when possible, then apportion extras."""
    groups = len(counts)
    if not groups or any(count <= 0 for count in counts) or not 0 < keep <= sum(counts):
        raise ValueError("Invalid temporal token counts or retention budget")
    quotas = [0] * groups
    if keep < groups:
        for index in _even_indices(groups, keep, torch.device("cpu")).tolist():
            quotas[index] = 1
        return quotas
    quotas = [1] * groups
    remaining = keep - groups
    if remaining == 0:
        return quotas
    capacities = [count - 1 for count in counts]
    total_capacity = sum(capacities)
    extras = [remaining * capacity // total_capacity for capacity in capacities]
    remainders = [remaining * capacity % total_capacity for capacity in capacities]
    for index, extra in enumerate(extras):
        quotas[index] += extra
    leftover = remaining - sum(extras)
    for index in sorted(range(groups), key=lambda i: (-remainders[i], i)):
        if leftover == 0:
            break
        if quotas[index] < counts[index]:
            quotas[index] += 1
            leftover -= 1
    if leftover or sum(quotas) != keep:
        raise RuntimeError("Could not allocate the uniform token budget")
    return quotas


def _morton_order(height: torch.Tensor, width: torch.Tensor) -> torch.Tensor:
    """Order a spatial patch grid so equal rank intervals cover 2D regions."""
    height = height.long() - height.min()
    width = width.long() - width.min()
    bits = max(int(height.max().item()), int(width.max().item())).bit_length()
    code = torch.zeros_like(height)
    for bit in range(bits):
        code |= ((width >> bit) & 1) << (2 * bit)
        code |= ((height >> bit) & 1) << (2 * bit + 1)
    return torch.argsort(code, stable=True)


def build_uniform_prune_sequence_indices(
    inputs_embeds: torch.Tensor,
    video_mask: torch.Tensor,
    position_ids: torch.Tensor,
    retain_ratio: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Keep an exact visual-token budget across mRoPE time and spatial grids.

    Groups are Qwen's temporal position IDs, which can represent temporal
    patches rather than individual decoded video frames. Within each group,
    equally spaced Morton-order ranks provide deterministic spatial coverage.
    """
    if inputs_embeds.ndim != 3 or inputs_embeds.shape[0] != 1:
        raise ValueError("Uniform pruning requires one [1, sequence, hidden] input")
    if video_mask.shape != inputs_embeds.shape[:2]:
        raise ValueError("Video mask does not match projected embeddings")
    if position_ids.ndim != 3 or position_ids.shape[:2] != (3, 1):
        raise ValueError("Expected Qwen mRoPE position IDs shaped [3, 1, sequence]")
    if position_ids.shape[-1] != inputs_embeds.shape[1]:
        raise ValueError("mRoPE positions do not match projected embeddings")

    mask = video_mask.to(device=inputs_embeds.device, dtype=torch.bool)
    positions = position_ids.to(inputs_embeds.device)
    visual_positions = torch.nonzero(mask[0], as_tuple=False).squeeze(1)
    if visual_positions.numel() == 0:
        raise ValueError("No projected video tokens were found")
    budget = retained_token_count(int(visual_positions.numel()), retain_ratio)
    if budget == len(visual_positions):
        return torch.arange(inputs_embeds.shape[1], device=inputs_embeds.device), visual_positions

    temporal = positions[0, 0].index_select(0, visual_positions)
    group_ids, inverse = torch.unique(temporal, sorted=True, return_inverse=True)
    groups = [visual_positions[inverse == index] for index in range(len(group_ids))]
    quotas = _temporal_quotas([len(group) for group in groups], budget)
    selected = []
    for group, quota in zip(groups, quotas):
        if quota == 0:
            continue
        if quota == len(group):
            selected.append(group)
            continue
        spatial = positions[1:, 0].index_select(1, group)
        order = _morton_order(spatial[0], spatial[1])
        selected.append(group.index_select(0, order.index_select(
            0, _even_indices(len(group), quota, group.device)
        )))
    selected_visual = torch.cat(selected).sort().values
    if selected_visual.numel() != budget or selected_visual.unique().numel() != budget:
        raise RuntimeError("Uniform pruning did not select the exact unique token budget")
    keep_mask = ~mask[0]
    keep_mask[selected_visual] = True
    return torch.nonzero(keep_mask, as_tuple=False).squeeze(1), selected_visual
