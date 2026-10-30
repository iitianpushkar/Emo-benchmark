"""Capture Qwen2.5 vision attention received by merged video patches.

Compatible with the Qwen2.5-VL architecture in transformers 4.57.1. This
computes attention probabilities alongside the SDPA forward; it does not
replace the model's attention implementation or do a second vision forward.
"""

from __future__ import annotations

import torch


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    half = x.shape[-1] // 2
    return torch.cat((-x[..., half:], x[..., :half]), dim=-1)


def received_attention(qkv: torch.Tensor, position_embeddings: tuple[torch.Tensor, torch.Tensor],
                       cu_seqlens: torch.Tensor, heads: int) -> torch.Tensor:
    """Mean across heads and queries of attention received by each patch."""
    length = qkv.shape[0]
    q, k, _ = qkv.reshape(length, 3, heads, -1).permute(1, 0, 2, 3).unbind(0)
    cos, sin = position_embeddings
    cos, sin = cos.unsqueeze(1).float(), sin.unsqueeze(1).float()
    q = q.float() * cos + _rotate_half(q.float()) * sin
    k = k.float() * cos + _rotate_half(k.float()) * sin
    scaling = q.shape[-1] ** -0.5
    boundaries = cu_seqlens.tolist()
    scores = []
    for start, end in zip(boundaries[:-1], boundaries[1:]):
        query = q[start:end].transpose(0, 1)
        key = k[start:end].transpose(0, 1)
        attention = (query @ key.transpose(-1, -2) * scaling).softmax(dim=-1)
        scores.append(attention.mean(dim=(0, 1)))
    return torch.cat(scores)


class QwenVisionAttentionCapture:
    def __init__(self, model, video_grid_thw: torch.Tensor):
        visual = model.model.visual
        full_blocks = list(visual.fullatt_block_indexes)
        if not full_blocks:
            raise ValueError("Qwen vision encoder has no full-attention block")
        self.visual = visual
        self.attn = visual.blocks[full_blocks[-1]].attn
        self.grid = video_grid_thw
        self.scores = None
        self.handles = []
        self._position_embeddings = None
        self._cu_seqlens = None

    def __enter__(self):
        self.handles = [
            self.attn.register_forward_pre_hook(self._pre, with_kwargs=True),
            self.attn.qkv.register_forward_hook(self._qkv),
        ]
        return self

    def __exit__(self, *_):
        for handle in self.handles:
            handle.remove()
        self.handles = []
        self._position_embeddings = None
        self._cu_seqlens = None

    def _pre(self, _module, _args, kwargs):
        self._position_embeddings = kwargs["position_embeddings"]
        self._cu_seqlens = kwargs["cu_seqlens"]

    def _qkv(self, _module, _args, output):
        if self.scores is not None:
            raise RuntimeError("Vision attention was captured more than once")
        if self._position_embeddings is None or self._cu_seqlens is None:
            raise RuntimeError("Missing Qwen vision attention inputs")
        patch_scores = received_attention(output, self._position_embeddings,
                                          self._cu_seqlens, self.attn.num_heads)
        merge = self.visual.spatial_merge_unit
        if patch_scores.numel() % merge:
            raise ValueError("Vision attention score count is not divisible by merge unit")
        window_index, _ = self.visual.get_window_index(self.grid)
        reverse = torch.argsort(window_index.to(patch_scores.device))
        self.scores = patch_scores.reshape(-1, merge).mean(dim=1)[reverse]
