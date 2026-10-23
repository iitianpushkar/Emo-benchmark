"""Prune Qwen2.5-VL projected video tokens before language-model prefill."""

from __future__ import annotations

import torch

from scripts.pruning.divprune import build_divprune_sequence_indices
from scripts.pruning.random_prune import build_random_prune_sequence_indices


class DivPruneHook:
    """Preserve selected video embeddings and their original multimodal positions."""

    def __init__(self, model, input_ids: torch.Tensor, retain_ratio: float,
                 method: str = "divprune", random_seed: int | None = None):
        if not 0 < retain_ratio <= 1:
            raise ValueError("retain_ratio must be in (0, 1]")
        if method not in ("divprune", "random"):
            raise ValueError(f"Unknown token pruning method: {method}")
        if method == "random" and random_seed is None:
            raise ValueError("Random token pruning requires a per-sample seed")
        self.method = method
        self.random_seed = random_seed
        self.retain_ratio = retain_ratio
        self.video_mask = input_ids.eq(model.config.video_token_id)
        self.full_tokens = int(self.video_mask.sum().item())
        self.full_sequence_tokens = int(input_ids.shape[-1])
        if self.full_tokens == 0:
            raise ValueError("Qwen2.5 input contains no expanded video tokens")
        self.retained_tokens = self.full_tokens
        self.retained_sequence_tokens = self.full_sequence_tokens
        self.calls = 0
        self.handle = None
        self.language_model = model.model.language_model

    def __enter__(self):
        if self.retain_ratio < 1:
            self.handle = self.language_model.register_forward_pre_hook(self._prune, with_kwargs=True)
        return self

    def __exit__(self, exc_type, exc, traceback):
        if self.handle is not None:
            self.handle.remove()
        if exc_type is None and self.retain_ratio < 1 and self.calls != 1:
            raise RuntimeError(f"Expected one Qwen2.5 language-model call, observed {self.calls}")

    def _prune(self, _module, args, kwargs):
        self.calls += 1
        embeds = kwargs.get("inputs_embeds")
        if not isinstance(embeds, torch.Tensor) or embeds.ndim != 3 or embeds.shape[0] != 1:
            raise ValueError("Token pruning requires one Qwen2.5 input with 3D projected embeddings")
        video_mask = self.video_mask.to(embeds.device)
        if video_mask.shape != embeds.shape[:2]:
            raise ValueError("Expanded video mask does not match projected embeddings")
        if self.method == "random":
            keep, selected = build_random_prune_sequence_indices(
                embeds, video_mask, self.retain_ratio, seed=self.random_seed,
            )
        else:
            keep, selected = build_divprune_sequence_indices(embeds, video_mask, self.retain_ratio)
        self.retained_tokens = int(selected.numel())
        self.retained_sequence_tokens = int(keep.numel())

        mask = kwargs.get("attention_mask")
        positions = kwargs.get("position_ids")
        if not isinstance(mask, torch.Tensor) or mask.shape[-1] != embeds.shape[1]:
            raise ValueError("Qwen2.5 attention mask does not match the token sequence")
        if not isinstance(positions, torch.Tensor) or positions.shape[-1] != embeds.shape[1]:
            raise ValueError("Qwen2.5 multimodal position IDs do not match the token sequence")
        if kwargs.get("cache_position") is not None:
            raise ValueError("Token pruning only supports uncached Qwen2.5 embedding extraction")

        updated = dict(kwargs)
        updated["inputs_embeds"] = embeds.index_select(1, keep)
        updated["attention_mask"] = mask.index_select(mask.ndim - 1, keep.to(mask.device))
        updated["position_ids"] = positions.index_select(positions.ndim - 1, keep.to(positions.device))
        return args, updated
