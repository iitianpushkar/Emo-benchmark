"""SparseVLM v1.5-style progressive text-guided pruning for Qwen2.5-VL.

Reference: Gumpest/SparseVLMs, v1.5, score.py and modelling_sparse_llama.py.
The paper also reports Qwen2-VL benchmarks, while the public v1.5 source
used here implements the LLaVA path. This Qwen2.5-VL adaptation keeps its
text-rater attention ranking, rank-based stage reductions, three pruning
stages, and 30%-then-cluster recycling, while using exact final budgets and
Qwen multimodal positions.
Only batch-one, uncached embedding extraction is supported.
"""

from __future__ import annotations

import math

import torch

from scripts.pruning.divprune import retained_token_count

DEFAULT_STAGES = (2, 6, 15)  # zero-based decoder layers, as in official v1.5


def rank_adaptive_target(current: int, rank: int, final: int, lam: float,
                         last_stage: bool) -> tuple[int, int]:
    """Paper N=floor(lambda*(L_v-rank(P))), capped by a matched final budget."""
    if not 1 <= final <= current or not 0 <= rank <= current or not 0 <= lam <= 1:
        raise ValueError("Invalid SparseVLM rank/budget/lambda")
    proposed = math.floor(lam * (current - rank))
    target = final if last_stage else max(final, current - proposed)
    return target, proposed


def recycle_count(discarded: int) -> int:
    """Official v1.5: top 30% of discarded tokens, then about one in ten centers."""
    if discarded <= 0:
        return 0
    candidates = min(discarded, int(discarded * 0.3) + 1)
    return min(candidates, candidates // 10 + 1)


def direct_keep_count(current: int, target: int) -> tuple[int, int]:
    """Allocate kept and recycled tokens without exceeding the target."""
    if not 1 <= target <= current:
        raise ValueError("Invalid stage target")
    if target == current:
        return current, 0
    for direct in range(target, 0, -1):
        merged = recycle_count(current - direct)
        if direct + merged == target:
            return direct, merged
    # Very small targets may leave no room for a recycled token.
    return target, 0


def choose_text_raters(hidden: torch.Tensor, video_mask: torch.Tensor) -> torch.Tensor:
    """Official embedding-similarity rule, restricted to text after video."""
    visual = torch.where(video_mask)[0]
    text = torch.where(~video_mask & (torch.arange(video_mask.numel(), device=video_mask.device)
                                      > visual[-1]))[0]
    if text.numel() == 0:
        raise ValueError("SparseVLM needs text tokens after the video")
    similarity = hidden[0, visual].float() @ hidden[0, text].float().T
    relevance = similarity.softmax(dim=-1).mean(dim=0)
    selected = text[relevance > relevance.mean()]
    return selected if selected.numel() else text[relevance.argmax().view(1)]


def text_visual_attention(q_projection: torch.Tensor, k_projection: torch.Tensor,
                          position_embeddings: tuple[torch.Tensor, torch.Tensor],
                          mrope_section: list[int], heads: int, key_value_heads: int,
                          attention_mask: torch.Tensor | None, raters: torch.Tensor,
                          return_matrix: bool = False) -> torch.Tensor:
    """Selected text-query attention; optionally return P for matrix rank."""
    from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import apply_multimodal_rotary_pos_emb

    batch, length, _ = q_projection.shape
    if batch != 1 or k_projection.shape[:2] != (1, length):
        raise ValueError("SparseVLM supports one uncached sequence")
    dim = q_projection.shape[-1] // heads
    q = q_projection.reshape(1, length, heads, dim).transpose(1, 2)
    k = k_projection.reshape(1, length, key_value_heads, dim).transpose(1, 2)
    q, k = apply_multimodal_rotary_pos_emb(q, k, *position_embeddings, mrope_section)
    if heads % key_value_heads:
        raise ValueError("Qwen attention heads are incompatible with GQA")
    k = k.repeat_interleave(heads // key_value_heads, dim=1)
    rows = raters.to(q.device)
    logits = (q.index_select(-2, rows).float() @ k.float().transpose(-1, -2)) * dim**-0.5
    if attention_mask is not None:
        if attention_mask.ndim != 4 or attention_mask.shape[-2:] != (length, length):
            raise ValueError("Unexpected Qwen attention-mask shape")
        logits = logits + attention_mask.index_select(-2, rows).float()
    matrix = logits.softmax(dim=-1).mean(dim=1)[0]
    return matrix if return_matrix else matrix.mean(dim=0)


def cluster_and_merge(features: torch.Tensor, centers: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Density-peaks centers and mean aggregation from official SparseVLM utils.py."""
    n, dim = features.shape
    if not 1 <= centers <= n:
        raise ValueError("Invalid recycling-center count")
    distances = torch.cdist(features.float().unsqueeze(0), features.float().unsqueeze(0))[0]
    distances = distances / math.sqrt(dim)
    nearest = distances.topk(centers, largest=False).values
    density = (-nearest.square().mean(dim=-1)).exp()
    higher = density.unsqueeze(0) > density.unsqueeze(1)
    farthest = distances.max()
    distance_to_higher = torch.where(higher, distances, farthest).min(dim=-1).values
    rank = (distance_to_higher * density).argsort(descending=True, stable=True)[:centers]
    assignments = distances.index_select(0, rank).argmin(dim=0)
    assignments[rank] = torch.arange(centers, device=features.device)
    merged = torch.zeros((centers, dim), dtype=features.dtype, device=features.device)
    merged.index_add_(0, assignments, features)
    counts = torch.bincount(assignments, minlength=centers).to(features.dtype).clamp_min(1)
    return rank, merged / counts[:, None]


class SparseVLMHook:
    """Apply progressive pruning/recycling after decoder layers 2, 6, and 15."""

    def __init__(self, model, input_ids: torch.Tensor, retain_ratio: float,
                 stages: tuple[int, ...] = DEFAULT_STAGES):
        if not 0 < retain_ratio <= 1 or input_ids.ndim != 2 or input_ids.shape[0] != 1:
            raise ValueError("SparseVLM requires batch one and retention in (0, 1]")
        self.layers = model.model.language_model.layers
        if not stages or tuple(sorted(set(stages))) != stages or stages[0] < 0 or stages[-1] >= len(self.layers) - 1:
            raise ValueError("SparseVLM stages must be ordered decoder layers before the final layer")
        self.stages = stages
        self.retain_ratio = retain_ratio
        self.video_mask = input_ids.eq(model.config.video_token_id)[0]
        self.full_tokens = int(self.video_mask.sum())
        self.full_sequence_tokens = int(input_ids.shape[-1])
        if self.full_tokens == 0:
            raise ValueError("Qwen input has no video tokens")
        self.retained_tokens = self.full_tokens
        self.retained_sequence_tokens = self.full_sequence_tokens
        self.final_budget = retained_token_count(self.full_tokens, retain_ratio)
        # With approximately full redundancy, three equal multiplicative cuts
        # would reach the requested final budget. Actual cuts use rank(P).
        self.lam = 1.0 - retain_ratio ** (1.0 / len(stages))
        self.active = torch.arange(self.full_sequence_tokens, device=input_ids.device)
        self.next_active = self.active
        self.raters = None
        self.replacements = {}
        self.scores = None
        self.priority = None
        self.priority_rank = None
        self._q = self._k = None
        self.handles = []
        self.stage_metadata = []
        self._completed_stages = 0
        self._layer_calls = 0

    def __enter__(self):
        if self.retain_ratio == 1:
            return self
        self.handles.append(self.layers[0].register_forward_pre_hook(self._select_raters, with_kwargs=True))
        for index, layer in enumerate(self.layers):
            if index:
                self.handles.append(layer.register_forward_pre_hook(self._prepare_layer, with_kwargs=True))
            if index in self.stages:
                attn = layer.self_attn
                self.handles.append(attn.q_proj.register_forward_hook(self._capture_q))
                self.handles.append(attn.k_proj.register_forward_hook(self._capture_k))
                self.handles.append(attn.register_forward_hook(self._score, with_kwargs=True))
                self.handles.append(layer.register_forward_hook(self._finish_stage))
        return self

    def __exit__(self, exc_type, _exc, _traceback):
        for handle in self.handles:
            handle.remove()
        self.handles.clear()
        self._q = self._k = None
        if exc_type is None and self.retain_ratio < 1:
            if self._completed_stages != len(self.stages) or self._layer_calls != len(self.layers) - 1:
                raise RuntimeError("SparseVLM stages/layers did not execute as expected")
            if self.retained_tokens != self.final_budget:
                raise RuntimeError("SparseVLM final visual-token budget changed")

    def _select_raters(self, _module, args, _kwargs):
        if self.raters is not None:
            raise RuntimeError("SparseVLM raters selected more than once")
        self.raters = choose_text_raters(args[0], self.video_mask.to(args[0].device))

    def _capture_q(self, _module, _args, output):
        self._q = output

    def _capture_k(self, _module, _args, output):
        self._k = output

    def _prepare_layer(self, _module, args, kwargs):
        self._layer_calls += 1
        if self.raters is None or self._layer_calls >= len(self.layers):
            raise RuntimeError("SparseVLM layer state is inconsistent")
        hidden = args[0]
        if hidden.shape[1] != self.active.numel():
            raise ValueError("SparseVLM received unexpected hidden-sequence length")
        if not torch.equal(self.active, self.next_active):
            positions = torch.searchsorted(self.active, self.next_active)
            if not torch.equal(self.active.index_select(0, positions), self.next_active):
                raise ValueError("SparseVLM next positions are not a subset of current positions")
            hidden = hidden.index_select(1, positions.to(hidden.device))
            for absolute, feature in self.replacements.items():
                relative = torch.searchsorted(self.next_active, torch.tensor(absolute, device=self.next_active.device))
                hidden[0, relative] = feature
            self.replacements.clear()
            self.active = self.next_active
        updated = dict(kwargs)
        original = self.full_sequence_tokens
        cos, sin = updated["position_embeddings"]
        if cos.shape[-2] != original or sin.shape[-2] != original:
            raise ValueError("SparseVLM requires full Qwen multimodal position embeddings")
        updated["position_embeddings"] = (cos.index_select(-2, self.active.to(cos.device)),
                                           sin.index_select(-2, self.active.to(sin.device)))
        mask = updated.get("attention_mask")
        if mask is not None:
            if mask.ndim != 4 or mask.shape[-2:] != (original, original):
                raise ValueError("SparseVLM requires a full additive causal mask")
            position = self.active.to(mask.device)
            updated["attention_mask"] = mask.index_select(-2, position).index_select(-1, position)
        for key in ("position_ids", "cache_position"):
            value = updated.get(key)
            if value is not None:
                updated[key] = value.index_select(-1, self.active.to(value.device))
        if updated.get("past_key_values") is not None or updated.get("use_cache"):
            raise ValueError("SparseVLM requires use_cache=False")
        return (hidden,) + args[1:], updated

    def _score(self, module, _args, kwargs, _output):
        if self._q is None or self._k is None or self.raters is None:
            raise RuntimeError("SparseVLM did not capture Qwen attention projections")
        raters = torch.searchsorted(self.active, self.raters)
        if not torch.equal(self.active.index_select(0, raters), self.raters):
            raise ValueError("SparseVLM removed a text rater")
        matrix = text_visual_attention(
            self._q, self._k, kwargs["position_embeddings"],
            module.rope_scaling["mrope_section"], module.num_heads,
            module.num_key_value_heads, kwargs.get("attention_mask"), raters,
            return_matrix=True,
        )
        visual_relative = torch.where(self.video_mask.to(self.active.device)[self.active])[0]
        self.priority = matrix.index_select(-1, visual_relative)
        self.priority_rank = int(torch.linalg.matrix_rank(self.priority.float()).item())
        self.scores = matrix.mean(dim=0)
        self._q = self._k = None

    def _finish_stage(self, _module, _args, output):
        stage = self._completed_stages
        if self.scores is None or stage >= len(self.stages):
            raise RuntimeError("SparseVLM stage did not receive attention scores")
        hidden = output[0] if isinstance(output, tuple) else output
        visual_relative = torch.where(self.video_mask.to(self.active.device)[self.active])[0]
        current = int(visual_relative.numel())
        target, proposed_deletions = rank_adaptive_target(
            current, self.priority_rank, self.final_budget, self.lam,
            last_stage=stage == len(self.stages) - 1,
        )
        direct, merged_count = direct_keep_count(current, target)
        rank = self.scores.index_select(0, visual_relative).argsort(descending=True, stable=True)
        retained_relative = visual_relative.index_select(0, rank[:direct])
        dropped_relative = visual_relative.index_select(0, rank[direct:])
        recycled_relative = torch.empty(0, dtype=torch.long, device=self.active.device)
        if merged_count:
            candidates = min(dropped_relative.numel(), int(dropped_relative.numel() * 0.3) + 1)
            chosen = dropped_relative[:candidates]
            centers, features = cluster_and_merge(hidden[0].index_select(0, chosen), merged_count)
            recycled_relative = chosen.index_select(0, centers)
            for absolute, feature in zip(self.active.index_select(0, recycled_relative).tolist(), features):
                self.replacements[absolute] = feature
        selected_relative = torch.cat((retained_relative, recycled_relative))
        keep = ~self.video_mask.to(self.active.device)[self.active]
        keep[selected_relative] = True
        self.next_active = self.active[keep]
        if self.next_active.numel() != self.full_sequence_tokens - self.full_tokens + target:
            raise RuntimeError("SparseVLM stage token budget mismatch")
        if self.next_active[-1] != self.full_sequence_tokens - 1:
            raise RuntimeError("SparseVLM lost the last prompt token")
        self.retained_tokens = target
        self.retained_sequence_tokens = int(self.next_active.numel())
        self.stage_metadata.append({"layer": self.stages[stage], "input_visual_tokens": current,
                                    "priority_matrix_rank": self.priority_rank,
                                    "priority_matrix_shape": list(self.priority.shape),
                                    "lambda": self.lam,
                                    "rank_rule_proposed_deletions": proposed_deletions,
                                    "applied_deletions_before_recycling": current - direct,
                                    "matched_budget_adjustment": (current - target) != proposed_deletions,
                                    "direct_kept_visual_tokens": direct,
                                    "recycled_visual_tokens": merged_count,
                                    "output_visual_tokens": target,
                                    "recycled_original_positions": self.active.index_select(0, recycled_relative).tolist()})
        self._completed_stages += 1
        self.scores = None
        self.priority = None
        self.priority_rank = None
