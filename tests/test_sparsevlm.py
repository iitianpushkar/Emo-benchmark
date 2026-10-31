"""Unit checks for the Qwen SparseVLM adaptation's budgets and sequence edits."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cluster.qwen_sparsevlm import (  # noqa: E402
    SparseVLMHook, choose_text_raters, cluster_and_merge, direct_keep_count,
    rank_adaptive_target, text_visual_attention,
)


class SparseVLMTests(unittest.TestCase):
    def test_rank_changes_stage_budget_and_final_budget_is_exact(self):
        self.assertEqual(rank_adaptive_target(100, 10, 15, 0.5, False), (55, 45))
        self.assertEqual(rank_adaptive_target(100, 80, 15, 0.5, False), (90, 10))
        self.assertEqual(rank_adaptive_target(100, 10, 15, 0.5, True), (15, 45))
        self.assertEqual(rank_adaptive_target(20, 0, 15, 0.5, False), (15, 10))
        kept, recycled = direct_keep_count(100, 55)
        self.assertEqual(kept + recycled, 55)
        self.assertEqual(direct_keep_count(2, 1), (1, 0))

    def test_text_raters_exclude_interleaved_video_and_prefix(self):
        hidden = torch.zeros((1, 7, 2))
        hidden[0, [1, 3, 4], 0] = 1
        hidden[0, 5, 0] = 2
        hidden[0, 6, 1] = 2
        mask = torch.tensor([False, True, False, True, True, False, False])
        self.assertEqual(choose_text_raters(hidden, mask).tolist(), [5])

    def test_recycling_centers_and_averages(self):
        features = torch.tensor([[0., 0.], [0.2, 0.], [10., 0.], [10.2, 0.]])
        centers, merged = cluster_and_merge(features, 2)
        self.assertEqual(len(set(centers.tolist())), 2)
        self.assertEqual(merged.shape, (2, 2))
        self.assertTrue(torch.allclose(merged[:, 0].sort().values, torch.tensor([0.1, 10.1]), atol=1e-4))

    def test_text_attention_uses_qwen_rope_and_mask(self):
        q = torch.zeros((1, 4, 24))
        k = torch.zeros((1, 4, 12))
        q[0, 3, 0] = 2
        q[0, 3, 12] = 2
        k[0, 1, 0] = 1
        k[0, 2, 0] = 3
        cos = torch.ones((3, 1, 4, 12))
        sin = torch.zeros_like(cos)
        mask = torch.triu(torch.full((4, 4), -1e4), diagonal=1).reshape(1, 1, 4, 4)
        scores = text_visual_attention(q, k, (cos, sin), [2, 2, 2], 2, 1,
                                       mask, torch.tensor([3]))
        matrix = text_visual_attention(q, k, (cos, sin), [2, 2, 2], 2, 1,
                                       mask, torch.tensor([3]), return_matrix=True)
        self.assertEqual(scores.shape, (4,))
        self.assertEqual(matrix.shape, (1, 4))
        self.assertTrue(torch.allclose(matrix.mean(0), scores))
        self.assertEqual(int(scores.argmax()), 2)

    def test_recycled_positions_preserve_causal_order(self):
        class Layer:
            pass

        layers = [Layer() for _ in range(17)]
        model = SimpleNamespace(model=SimpleNamespace(language_model=SimpleNamespace(layers=layers)),
                                config=SimpleNamespace(video_token_id=99))
        hook = SparseVLMHook(model, torch.tensor([[1, 99, 99, 99, 2, 3]]), 1 / 3)
        hook.raters = torch.tensor([4, 5])
        hook.active = torch.arange(6)
        hook.next_active = torch.tensor([0, 1, 3, 4, 5])
        hook.replacements = {3: torch.tensor([9., 9.])}
        hidden = torch.arange(12, dtype=torch.float32).reshape(1, 6, 2)
        causal = torch.triu(torch.full((6, 6), -1e4), diagonal=1).reshape(1, 1, 6, 6)
        cos = torch.ones((3, 1, 6, 2))
        kwargs = {"attention_mask": causal, "position_embeddings": (cos, torch.zeros_like(cos)),
                  "cache_position": torch.arange(6), "use_cache": False}
        args, updated = hook._prepare_layer(layers[1], (hidden,), kwargs)
        self.assertEqual(args[0].shape, (1, 5, 2))
        self.assertEqual(args[0][0, 2].tolist(), [9., 9.])
        self.assertEqual(updated["attention_mask"].shape, (1, 1, 5, 5))
        self.assertEqual(updated["cache_position"].tolist(), [0, 1, 3, 4, 5])


if __name__ == "__main__":
    unittest.main()
