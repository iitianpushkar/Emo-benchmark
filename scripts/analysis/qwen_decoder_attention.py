"""Capture Qwen2.5-VL decoder attention from the final prompt-token query.

The model can keep SDPA enabled. Hooks retain one layer's Q/K projections only
until that attention module returns, reconstruct the final causal query row,
move the head-averaged result to CPU, and release the projections. This avoids
materializing every full sequence-by-sequence attention matrix.
"""

from __future__ import annotations

import torch


def last_query_attention_per_head(
    q_projection: torch.Tensor,
    k_projection: torch.Tensor,
    position_embeddings: tuple[torch.Tensor, torch.Tensor],
    mrope_section: list[int],
    heads: int,
    key_value_heads: int,
    attention_mask: torch.Tensor | None,
) -> torch.Tensor:
    """Return post-RoPE attention from the final query, shaped ``[heads, keys]``."""
    from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import (
        apply_multimodal_rotary_pos_emb,
    )

    batch, length, width = q_projection.shape
    if batch != 1 or k_projection.shape[:2] != (1, length):
        raise ValueError("Decoder attention capture requires one uncached sequence")
    if width % heads:
        raise ValueError("Query width is not divisible by the attention-head count")
    head_dim = width // heads
    q = q_projection.reshape(1, length, heads, head_dim).transpose(1, 2)
    k = k_projection.reshape(1, length, key_value_heads, head_dim).transpose(1, 2)
    cos, sin = position_embeddings
    q, k = apply_multimodal_rotary_pos_emb(q, k, cos, sin, mrope_section)
    if heads % key_value_heads:
        raise ValueError("Attention heads must be divisible by key/value heads")
    k = k.repeat_interleave(heads // key_value_heads, dim=1)
    logits = (q[:, :, -1:, :].float() @ k.float().transpose(-1, -2)) * head_dim**-0.5
    if attention_mask is not None:
        if attention_mask.ndim != 4 or attention_mask.shape[-1] != length:
            raise ValueError(f"Unexpected Qwen attention-mask shape: {tuple(attention_mask.shape)}")
        logits = logits + attention_mask[:, :, -1:, :].float()
    return logits.softmax(dim=-1)[0, :, 0]


class QwenLastQueryAttentionCapture:
    """Capture mean final-query attention at every language-decoder layer."""

    def __init__(self, model, sequence_length: int):
        self.layers = model.model.language_model.layers
        self.sequence_length = int(sequence_length)
        if self.sequence_length <= 0:
            raise ValueError("sequence_length must be positive")
        self.scores: dict[int, torch.Tensor] = {}
        self._projections: dict[int, dict[str, torch.Tensor]] = {}
        self._handles = []

    def __enter__(self):
        for index, layer in enumerate(self.layers):
            attention = layer.self_attn
            self._handles.extend(
                (
                    attention.q_proj.register_forward_hook(self._projection_hook(index, "q")),
                    attention.k_proj.register_forward_hook(self._projection_hook(index, "k")),
                    attention.register_forward_hook(
                        self._attention_hook(index), with_kwargs=True
                    ),
                )
            )
        return self

    def __exit__(self, exc_type, _exc, _traceback):
        for handle in self._handles:
            handle.remove()
        self._handles.clear()
        self._projections.clear()
        if exc_type is None and len(self.scores) != len(self.layers):
            missing = sorted(set(range(len(self.layers))) - set(self.scores))
            raise RuntimeError(f"Attention capture missed decoder layers: {missing}")

    def _projection_hook(self, index: int, name: str):
        def capture(_module, _inputs, output):
            self._projections.setdefault(index, {})[name] = output

        return capture

    def _attention_hook(self, index: int):
        def capture(module, _args, kwargs, _output):
            projections = self._projections.pop(index, {})
            if set(projections) != {"q", "k"}:
                raise RuntimeError(f"Layer {index} did not expose both Q and K projections")
            scores = last_query_attention_per_head(
                projections["q"],
                projections["k"],
                kwargs["position_embeddings"],
                module.rope_scaling["mrope_section"],
                module.num_heads,
                module.num_key_value_heads,
                kwargs.get("attention_mask"),
            )
            if scores.shape[-1] != self.sequence_length:
                raise ValueError(
                    f"Layer {index} returned {scores.shape[-1]} keys; "
                    f"expected {self.sequence_length}"
                )
            mean_scores = scores.mean(dim=0).detach().to(device="cpu", dtype=torch.float32)
            if not torch.isfinite(mean_scores).all():
                raise ValueError(f"Layer {index} produced non-finite attention")
            if not torch.allclose(mean_scores.sum(), torch.tensor(1.0), atol=2e-4, rtol=2e-4):
                raise ValueError(f"Layer {index} attention does not sum to one")
            self.scores[index] = mean_scores

        return capture

    def stacked(self) -> torch.Tensor:
        """Return captured head-mean attention as ``[layers, sequence]``."""
        if len(self.scores) != len(self.layers):
            raise RuntimeError("Run the model inside the capture context before reading scores")
        return torch.stack([self.scores[index] for index in range(len(self.layers))])
