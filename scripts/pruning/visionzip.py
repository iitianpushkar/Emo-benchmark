"""Qwen2.5-VL video adaptation of official VisionZip dominant/contextual selection.

Reference: https://github.com/JIA-Lab-research/VisionZip/blob/main/Qwen2_5_VL/qwen2_5vl_visionzip.py
The released Qwen path handles images. We allocate the matched clip budget by
temporal group and only merge within a group so distinct video moments do not
get averaged into a single token. This is an explicit video adaptation.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from scripts.pruning.divprune import retained_token_count


def _quotas(counts: list[int], budget: int, minimum: int = 0) -> list[int]:
    if budget < minimum * len(counts) or budget > sum(counts):
        raise ValueError("Infeasible VisionZip temporal budget")
    assigned = [minimum] * len(counts)
    capacity = [count - minimum for count in counts]
    remaining = budget - sum(assigned)
    if remaining:
        total = sum(capacity)
        shares = [remaining * count / total for count in capacity]
        for i, share in enumerate(shares):
            assigned[i] += min(capacity[i], int(share))
        while sum(assigned) < budget:
            choices = [i for i in range(len(counts)) if assigned[i] < counts[i]]
            i = max(choices, key=lambda j: (shares[j] - (assigned[j] - minimum), -j))
            assigned[i] += 1
    return assigned


def build_visionzip_sequence(
    inputs_embeds: torch.Tensor,
    video_mask: torch.Tensor,
    position_ids: torch.Tensor,
    attention_scores: torch.Tensor,
    vision_keys: torch.Tensor,
    retain_ratio: float,
    contextual_fraction: float = 1 / 14,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int, int]:
    """Return updated embeddings, kept sequence IDs, selected video IDs, and counts.

    The official Qwen code's 65:5 dominant/contextual ratio is scaled to each
    requested retention budget. Contextual targets are uniformly spaced among
    non-dominant tokens; other tokens join their nearest target by cosine key
    similarity, then their mean projected embedding is added to that target.
    """
    if inputs_embeds.ndim != 3 or inputs_embeds.shape[0] != 1:
        raise ValueError("VisionZip requires batch-one projected embeddings")
    if video_mask.shape != inputs_embeds.shape[:2] or position_ids.shape[-1] != inputs_embeds.shape[1]:
        raise ValueError("VisionZip masks and positions do not match Qwen sequence")
    video_sequence = torch.where(video_mask[0])[0]
    n = video_sequence.numel()
    if attention_scores.numel() != n or vision_keys.shape[0] != n:
        raise ValueError("VisionZip scores and keys must align with all video tokens")
    target = retained_token_count(n, retain_ratio)
    if target == n:
        return inputs_embeds, torch.arange(inputs_embeds.shape[1], device=inputs_embeds.device), torch.arange(n, device=inputs_embeds.device), n, 0
    temporal = position_ids[0, 0].index_select(0, video_sequence.to(position_ids.device))
    unique_temporal, inverse = torch.unique_consecutive(temporal, return_inverse=True)
    if unique_temporal.numel() == 0:
        raise ValueError("VisionZip found no temporal groups")
    groups = [torch.where(inverse == i)[0].to(inputs_embeds.device) for i in range(unique_temporal.numel())]
    counts = [group.numel() for group in groups]
    group_budgets = _quotas(counts, target, minimum=1 if target >= len(groups) else 0)
    contextual_budget = min(max(1, round(target * contextual_fraction)), target - 1)
    context_capacity = [max(0, k - 1) for k in group_budgets]
    contextual_budget = min(contextual_budget, sum(context_capacity))
    context_budgets = _quotas(context_capacity, contextual_budget)
    updated = inputs_embeds.clone()
    keep_video = []
    dominant_total = contextual_total = 0
    scores = attention_scores.to(inputs_embeds.device)
    keys = F.normalize(vision_keys.to(inputs_embeds.device).float(), dim=-1, eps=1e-8)
    for group, k, context_k in zip(groups, group_budgets, context_budgets):
        if k == 0:
            continue
        if k == group.numel():
            keep_video.append(group)
            dominant_total += k
            continue
        dominant_k = k - context_k
        dominant_local = torch.topk(scores[group], dominant_k).indices
        dominant = group[dominant_local]
        dominant_total += dominant_k
        if context_k:
            residual_mask = torch.ones(group.numel(), dtype=torch.bool, device=group.device)
            residual_mask[dominant_local] = False
            residual = group[residual_mask]
            # Uniformly spaced contextual targets in the residual sequence.
            target_local = torch.div(
                torch.arange(context_k, device=group.device) * residual.numel(),
                context_k, rounding_mode="floor",
            )
            context = residual[target_local]
            other_mask = torch.ones(residual.numel(), dtype=torch.bool, device=group.device)
            other_mask[target_local] = False
            to_merge = residual[other_mask]
            if to_merge.numel():
                nearest = (keys[to_merge] @ keys[context].T).argmax(dim=-1)
                source_embeddings = inputs_embeds[0, video_sequence[to_merge]]
                sums = torch.zeros((context_k, inputs_embeds.shape[-1]), device=updated.device, dtype=updated.dtype)
                sums.index_add_(0, nearest, source_embeddings)
                merged_counts = torch.bincount(nearest, minlength=context_k).clamp_min(1)
                updated[0, video_sequence[context]] = (
                    inputs_embeds[0, video_sequence[context]]
                    + sums / merged_counts[:, None].to(sums.dtype)
                )
            keep_video.append(torch.cat((dominant, context)))
            contextual_total += context_k
        else:
            keep_video.append(dominant)
    selected = torch.cat(keep_video).sort().values
    if selected.numel() != target:
        raise RuntimeError(f"VisionZip retained {selected.numel()} instead of {target} video tokens")
    keep_mask = ~video_mask[0].clone()
    keep_mask[video_sequence[selected]] = True
    keep = torch.where(keep_mask)[0]
    return updated, keep, selected, dominant_total, contextual_total
