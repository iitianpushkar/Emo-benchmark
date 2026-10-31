"""FastV's last-query attention ranking inside Qwen2.5-VL's language model.

The official FastV LLaVA implementation ranks visual tokens by the attention
they receive from the final prompt token after K decoder layers, then physically
drops low-ranked visual tokens before layer K (zero-based). This hook adapts
that rule to Qwen2.5-VL's video token mask and multimodal RoPE positions.
Only uncached, batch-one embedding extraction is supported.
"""

from __future__ import annotations

import torch

from scripts.pruning.divprune import retained_token_count


def select_fastv_indices(scores: torch.Tensor, video_mask: torch.Tensor,
                         retain_ratio: float) -> tuple[torch.Tensor, torch.Tensor]:
    """Keep top-attended visual positions and all nonvisual positions."""
    if scores.ndim != 1 or video_mask.ndim != 1 or scores.shape != video_mask.shape:
        raise ValueError("scores and video_mask must be matching 1D tensors")
    if not torch.isfinite(scores).all():
        raise ValueError("FastV attention scores must be finite")
    visual = torch.nonzero(video_mask.bool(), as_tuple=False).squeeze(1)
    if visual.numel() == 0:
        raise ValueError("FastV found no visual tokens")
    target = retained_token_count(int(visual.numel()), retain_ratio)
    ranking = torch.argsort(scores[visual].float(), descending=True, stable=True)
    selected = visual[ranking[:target]].sort().values
    keep_mask = ~video_mask.bool()
    keep_mask[selected] = True
    keep = torch.nonzero(keep_mask, as_tuple=False).squeeze(1)
    return keep, selected


def last_query_attention(q_projection: torch.Tensor, k_projection: torch.Tensor,
                         position_embeddings: tuple[torch.Tensor, torch.Tensor],
                         mrope_section: list[int], heads: int, key_value_heads: int,
                         attention_mask: torch.Tensor | None) -> torch.Tensor:
    """Reconstruct the target layer's final query attention under SDPA."""
    from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import apply_multimodal_rotary_pos_emb

    batch, length, _ = q_projection.shape
    if batch != 1 or k_projection.shape[:2] != (1, length):
        raise ValueError("FastV supports one uncached sequence")
    head_dim = q_projection.shape[-1] // heads
    q = q_projection.reshape(1, length, heads, head_dim).transpose(1, 2)
    k = k_projection.reshape(1, length, key_value_heads, head_dim).transpose(1, 2)
    cos, sin = position_embeddings
    q, k = apply_multimodal_rotary_pos_emb(q, k, cos, sin, mrope_section)
    if heads % key_value_heads:
        raise ValueError("Attention heads must be divisible by key/value heads")
    k = k.repeat_interleave(heads // key_value_heads, dim=1)
    logits = (q[:, :, -1:, :].float() @ k.float().transpose(-1, -2)) * head_dim**-0.5
    if attention_mask is not None:
        if attention_mask.ndim != 4 or attention_mask.shape[-2:] != (length, length):
            raise ValueError("FastV expects the Qwen additive causal attention mask")
        if not torch.is_floating_point(attention_mask):
            raise ValueError("FastV requires an additive floating-point causal mask")
        logits = logits + attention_mask[:, :, -1:, :].float()
    return logits.softmax(dim=-1).mean(dim=1)[0, 0]


def slice_layer_inputs(args, kwargs: dict, keep: torch.Tensor, prune_hidden: bool):
    """Slice Qwen layer inputs while preserving the original mRoPE coordinates."""
    if not args or not isinstance(args[0], torch.Tensor):
        raise ValueError("Expected hidden states as first decoder-layer argument")
    hidden = args[0]
    full_length = int(kwargs["position_embeddings"][0].shape[-2])
    if prune_hidden:
        if hidden.shape[1] != full_length:
            raise ValueError("First FastV pruning layer received an unexpected sequence length")
        args = (hidden.index_select(1, keep.to(hidden.device)),) + args[1:]
    elif hidden.shape[1] != keep.numel():
        raise ValueError("Subsequent FastV layer received an unexpected sequence length")
    updated = dict(kwargs)
    mask = updated.get("attention_mask")
    if mask is not None:
        if mask.ndim != 4 or mask.shape[-2:] != (full_length, full_length):
            raise ValueError("Unexpected Qwen causal-mask shape during FastV pruning")
        index = keep.to(mask.device)
        updated["attention_mask"] = mask.index_select(-2, index).index_select(-1, index)
    cos, sin = updated["position_embeddings"]
    updated["position_embeddings"] = (
        cos.index_select(-2, keep.to(cos.device)),
        sin.index_select(-2, keep.to(sin.device)),
    )
    for key in ("position_ids", "cache_position"):
        value = updated.get(key)
        if value is not None:
            updated[key] = value.index_select(-1, keep.to(value.device))
    if updated.get("past_key_values") is not None or updated.get("use_cache"):
        raise ValueError("FastV embedding extraction requires use_cache=False")
    return args, updated


class FastVHook:
    """Prune after K decoder layers using the preceding layer's last query."""

    def __init__(self, model, input_ids: torch.Tensor, retain_ratio: float, k: int = 2):
        if not 0 < retain_ratio <= 1:
            raise ValueError("retain_ratio must be in (0, 1]")
        self.layers = model.model.language_model.layers
        if not 1 <= k < len(self.layers):
            raise ValueError(f"FastV K must be in [1, {len(self.layers) - 1}]")
        if input_ids.ndim != 2 or input_ids.shape[0] != 1:
            raise ValueError("FastV requires batch size one")
        self.k = k
        self.retain_ratio = retain_ratio
        self.video_mask = input_ids.eq(model.config.video_token_id)[0]
        self.full_tokens = int(self.video_mask.sum().item())
        self.full_sequence_tokens = int(input_ids.shape[-1])
        if self.full_tokens == 0:
            raise ValueError("Qwen input contains no video tokens")
        self.retained_tokens = self.full_tokens
        self.retained_sequence_tokens = self.full_sequence_tokens
        self.selected_visual_positions = None
        self.keep = None
        self.handles = []
        self._q = None
        self._k = None
        self._score_calls = 0
        self._layer_calls = 0

    def __enter__(self):
        if self.retain_ratio < 1:
            scoring_attention = self.layers[self.k - 1].self_attn
            self.handles.append(scoring_attention.q_proj.register_forward_hook(self._capture_q))
            self.handles.append(scoring_attention.k_proj.register_forward_hook(self._capture_k))
            self.handles.append(scoring_attention.register_forward_hook(self._score, with_kwargs=True))
            for offset, layer in enumerate(self.layers[self.k:]):
                self.handles.append(layer.register_forward_pre_hook(
                    self._prune_first if offset == 0 else self._prune_later, with_kwargs=True,
                ))
        return self

    def __exit__(self, exc_type, _exc, _traceback):
        for handle in self.handles:
            handle.remove()
        self.handles = []
        self._q = self._k = None
        if exc_type is None and self.retain_ratio < 1:
            if self._score_calls != 1 or self._layer_calls != len(self.layers) - self.k:
                raise RuntimeError("FastV did not score and prune exactly once at the intended layers")

    def _capture_q(self, _module, _inputs, output):
        self._q = output

    def _capture_k(self, _module, _inputs, output):
        self._k = output

    def _score(self, module, _args, kwargs, output):
        self._score_calls += 1
        if self._q is None or self._k is None or self._score_calls != 1:
            raise RuntimeError("FastV failed to capture Qwen query/key projections")
        scores = last_query_attention(
            self._q, self._k, kwargs["position_embeddings"],
            module.rope_scaling["mrope_section"], module.num_heads,
            module.num_key_value_heads, kwargs.get("attention_mask"),
        )
        self._q = self._k = None
        self.keep, self.selected_visual_positions = select_fastv_indices(
            scores, self.video_mask.to(scores.device), self.retain_ratio,
        )
        self.retained_tokens = int(self.selected_visual_positions.numel())
        self.retained_sequence_tokens = int(self.keep.numel())
        if self.keep[-1].item() != self.full_sequence_tokens - 1:
            raise ValueError("FastV must preserve the final prompt token")

    def _prune_first(self, _module, args, kwargs):
        self._layer_calls += 1
        if self.keep is None:
            raise RuntimeError("FastV decoder layer ran before attention scoring")
        return slice_layer_inputs(args, kwargs, self.keep, prune_hidden=True)

    def _prune_later(self, _module, args, kwargs):
        self._layer_calls += 1
        if self.keep is None:
            raise RuntimeError("FastV decoder layer ran before attention scoring")
        return slice_layer_inputs(args, kwargs, self.keep, prune_hidden=False)
