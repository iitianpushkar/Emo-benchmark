"""Remove all projected video tokens at Qwen's language-model entry."""

from __future__ import annotations


class NoVisualTokensHook:
    """Keep text and special tokens, with their original multimodal positions."""

    def __init__(self, model, input_ids):
        import torch

        if input_ids.ndim != 2 or input_ids.shape[0] != 1:
            raise ValueError("No-visual pass requires one uncached sequence")
        video_mask = input_ids[0].eq(model.config.video_token_id)
        self.full_visual_tokens = int(video_mask.sum())
        if self.full_visual_tokens == 0:
            raise ValueError("Input contains no expanded video tokens")
        self.keep = torch.nonzero(~video_mask, as_tuple=False).squeeze(1)
        if self.keep.numel() == 0 or not bool(self.keep[-1] == input_ids.shape[-1] - 1):
            raise ValueError("The final prompt token must be retained")
        self.language_model = model.model.language_model
        self.handle = None
        self.calls = 0

    def __enter__(self):
        self.handle = self.language_model.register_forward_pre_hook(self._remove, with_kwargs=True)
        return self

    def __exit__(self, exc_type, _exc, _traceback):
        if self.handle is not None:
            self.handle.remove()
            self.handle = None
        if exc_type is None and self.calls != 1:
            raise RuntimeError(f"Expected one language-model forward, got {self.calls}")

    def _remove(self, _module, args, kwargs):
        import torch

        self.calls += 1
        if args:
            raise ValueError("Expected Qwen language-model inputs as keyword arguments")
        embeddings = kwargs.get("inputs_embeds")
        if not isinstance(embeddings, torch.Tensor) or embeddings.ndim != 3 or embeddings.shape[0] != 1:
            raise ValueError("Expected one 3D projected embedding sequence")
        original_length = embeddings.shape[1]
        if original_length != self.keep[-1].item() + 1:
            raise ValueError("Video mask and projected embeddings do not align")
        updated = dict(kwargs)
        updated["inputs_embeds"] = embeddings.index_select(1, self.keep.to(embeddings.device))
        mask = kwargs.get("attention_mask")
        if mask is not None:
            if mask.shape[-1] != original_length:
                raise ValueError("Attention mask and projected embeddings do not align")
            indices = self.keep.to(mask.device)
            if mask.ndim == 2:
                updated["attention_mask"] = mask.index_select(-1, indices)
            elif mask.ndim == 4 and mask.shape[-2] == original_length:
                updated["attention_mask"] = mask.index_select(-2, indices).index_select(-1, indices)
            else:
                raise ValueError("Unsupported Qwen attention mask shape")
        positions = kwargs.get("position_ids")
        if not isinstance(positions, torch.Tensor) or positions.shape[-1] != original_length:
            raise ValueError("Qwen multimodal position IDs are missing or misaligned")
        updated["position_ids"] = positions.index_select(-1, self.keep.to(positions.device))
        cache_position = kwargs.get("cache_position")
        if isinstance(cache_position, torch.Tensor) and cache_position.shape[-1] == original_length:
            updated["cache_position"] = cache_position.index_select(
                -1, self.keep.to(cache_position.device))
        if kwargs.get("past_key_values") is not None or kwargs.get("use_cache"):
            raise ValueError("No-visual comparison requires a fresh uncached forward")
        return args, updated
