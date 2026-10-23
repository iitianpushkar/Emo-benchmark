"""Prune Qwen3-VL projected video tokens at the language-model boundary."""

from __future__ import annotations

import torch

from scripts.pruning.divprune import build_divprune_sequence_indices


class DivPruneHook:
    """Keep Qwen3's RoPE positions and DeepStack features aligned with tokens."""

    def __init__(self, model, input_ids: torch.Tensor, retain_ratio: float):
        if not 0 < retain_ratio <= 1:
            raise ValueError("retain_ratio must be in (0, 1]")
        self.retain_ratio = retain_ratio
        self.video_mask = input_ids.eq(model.config.video_token_id)
        self.full_tokens = int(self.video_mask.sum().item())
        if self.full_tokens == 0:
            raise ValueError("Qwen3 input contains no expanded video tokens")
        self.retained_tokens = self.full_tokens
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
            raise RuntimeError(f"Expected one Qwen3 language-model call, observed {self.calls}")

    def _prune(self, _module, args, kwargs):
        self.calls += 1
        embeds = kwargs.get("inputs_embeds")
        if not isinstance(embeds, torch.Tensor) or embeds.ndim != 3 or embeds.shape[0] != 1:
            raise ValueError("DivPrune requires one Qwen3 input with 3D projected embeddings")
        video_mask = self.video_mask.to(embeds.device)
        if video_mask.shape != embeds.shape[:2]:
            raise ValueError("Expanded video mask does not match Qwen3 projected embeddings")
        keep, selected = build_divprune_sequence_indices(embeds, video_mask, self.retain_ratio)
        full_positions = torch.nonzero(video_mask[0], as_tuple=False).squeeze(1)
        selected_local = torch.searchsorted(full_positions, selected)
        self.retained_tokens = int(selected.numel())

        mask = kwargs.get("attention_mask")
        positions = kwargs.get("position_ids")
        visual_mask = kwargs.get("visual_pos_masks")
        deepstack = kwargs.get("deepstack_visual_embeds")
        if not isinstance(mask, torch.Tensor) or mask.shape[-1] != embeds.shape[1]:
            raise ValueError("Qwen3 attention mask does not match the token sequence")
        if not isinstance(positions, torch.Tensor) or positions.shape[-1] != embeds.shape[1]:
            raise ValueError("Qwen3 multimodal position IDs do not match the token sequence")
        if not isinstance(visual_mask, torch.Tensor) or not torch.equal(visual_mask.to(video_mask.device), video_mask):
            raise ValueError("Qwen3 DeepStack visual mask differs from expanded video positions")
        if not isinstance(deepstack, (list, tuple)) or not deepstack:
            raise ValueError("Qwen3 DeepStack features are unavailable")
        if kwargs.get("cache_position") is not None:
            raise ValueError("DivPrune only supports uncached Qwen3 embedding extraction")
        for features in deepstack:
            if not isinstance(features, torch.Tensor) or features.shape[0] != self.full_tokens:
                raise ValueError("Qwen3 DeepStack feature count differs from full video tokens")

        updated = dict(kwargs)
        updated["inputs_embeds"] = embeds.index_select(1, keep)
        updated["attention_mask"] = mask.index_select(mask.ndim - 1, keep.to(mask.device))
        updated["position_ids"] = positions.index_select(positions.ndim - 1, keep.to(positions.device))
        updated["visual_pos_masks"] = visual_mask.index_select(visual_mask.ndim - 1, keep.to(visual_mask.device))
        updated["deepstack_visual_embeds"] = [
            features.index_select(0, selected_local.to(features.device)) for features in deepstack
        ]
        return args, updated
