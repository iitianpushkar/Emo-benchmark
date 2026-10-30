"""VisPruner's importance-plus-iterative-deduplication rule for Qwen video.

The official LLaVA implementation uses CLS-to-patch attention from CLIP.
Qwen2.5-VL has no CLS token; callers provide mean attention received by each
merged visual token from its last full-attention vision block instead.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from scripts.pruning.divprune import retained_token_count

DEFAULT_IMPORTANT_RATIO = 0.5


def select_vispruner_token_indices(visual_tokens: torch.Tensor, attention_scores: torch.Tensor,
                                   retain_ratio: float, important_ratio: float = DEFAULT_IMPORTANT_RATIO) -> torch.Tensor:
    if visual_tokens.ndim != 2 or visual_tokens.shape[0] == 0:
        raise ValueError("visual_tokens must be [tokens, features] and nonempty")
    n = visual_tokens.shape[0]
    if attention_scores.shape != (n,):
        raise ValueError("attention_scores must have one value per visual token")
    if not torch.isfinite(visual_tokens).all() or not torch.isfinite(attention_scores).all():
        raise ValueError("visual tokens and attention scores must be finite")
    if not 0 <= important_ratio <= 1:
        raise ValueError("important_ratio must be in [0, 1]")
    target = retained_token_count(n, retain_ratio)
    if target == n:
        return torch.arange(n, device=visual_tokens.device)
    important_count = int(target * important_ratio)
    order = torch.argsort(attention_scores.float(), descending=True, stable=True)
    important = order[:important_count]
    residual = order[important_count:]
    diverse_count = target - important_count
    normalized = F.normalize(visual_tokens.float(), dim=-1, eps=1e-12)
    if diverse_count == 0:
        residual = residual[:0]
    while diverse_count > 0 and residual.numel() > diverse_count:
        first, second = residual[::2], residual[1::2]
        if second.numel() == 0:
            break
        similarity = normalized[first] @ normalized[second].T
        duplicates = similarity.max(dim=1).values
        remove = min(8, residual.numel() - diverse_count, first.numel())
        # Official VisPruner drops the highest-similarity tokens in the first
        # (even-indexed) partition, retaining the second partition intact.
        distinct = torch.argsort(duplicates, descending=True, stable=True)[remove:]
        residual = torch.cat((first[distinct], second))
    selected = torch.cat((important, residual))
    if selected.numel() != target:
        raise RuntimeError(f"VisPruner selected {selected.numel()} tokens instead of {target}")
    return selected.sort().values.long()


def build_vispruner_sequence_indices(inputs_embeds: torch.Tensor, video_mask: torch.Tensor,
                                     attention_scores: torch.Tensor, retain_ratio: float,
                                     important_ratio: float = DEFAULT_IMPORTANT_RATIO) -> tuple[torch.Tensor, torch.Tensor]:
    if inputs_embeds.ndim != 3 or inputs_embeds.shape[0] != 1 or video_mask.shape != inputs_embeds.shape[:2]:
        raise ValueError("Expected one input and a matching video mask")
    mask = video_mask.to(device=inputs_embeds.device, dtype=torch.bool)
    positions = torch.nonzero(mask[0], as_tuple=False).squeeze(1)
    selected_local = select_vispruner_token_indices(
        inputs_embeds[0, positions], attention_scores.to(inputs_embeds.device),
        retain_ratio, important_ratio,
    )
    selected = positions[selected_local]
    keep_mask = ~mask[0]
    keep_mask[selected] = True
    return torch.nonzero(keep_mask, as_tuple=False).squeeze(1), selected
