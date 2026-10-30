"""Qwen VisPruner selection and vision-attention alignment checks."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cluster.qwen_vispruner_attention import received_attention
from scripts.pruning.vispruner import build_vispruner_sequence_indices, select_vispruner_token_indices


class VisPrunerTests(unittest.TestCase):
    def test_exact_budget_and_attention_priority(self):
        tokens = torch.tensor([[1., 0.], [1., 0.], [0., 1.], [0., 1.], [-1., 0.], [0., -1.]])
        scores = torch.tensor([0., 0., 0., 0., 10., 9.])
        selected = select_vispruner_token_indices(tokens, scores, 0.5, important_ratio=2 / 3)
        self.assertEqual(len(selected), 3)
        self.assertTrue({4, 5}.issubset(set(selected.tolist())))
        self.assertEqual(selected.tolist(), sorted(set(selected.tolist())))

    def test_low_budget_and_odd_residual(self):
        tokens = torch.randn(11, 4)
        scores = torch.arange(11, dtype=torch.float)
        for ratio in (0.05, 0.1, 0.3, 0.7, 1.0):
            selected = select_vispruner_token_indices(tokens, scores, ratio)
            self.assertEqual(len(selected), max(1, round(11 * ratio)))
            self.assertEqual(selected.tolist(), sorted(set(selected.tolist())))
        all_important = select_vispruner_token_indices(tokens, scores, 0.3, important_ratio=1.0)
        self.assertEqual(all_important.tolist(), [8, 9, 10])

    def test_nonvisual_positions_are_kept(self):
        embeds = torch.randn(1, 8, 4)
        mask = torch.tensor([[False, True, True, True, True, True, True, False]])
        keep, selected = build_vispruner_sequence_indices(
            embeds, mask, torch.arange(6, dtype=torch.float), 0.5,
        )
        self.assertEqual(len(selected), 3)
        self.assertTrue({0, 7}.issubset(set(keep.tolist())))
        self.assertEqual(len(keep), 5)

    def test_received_attention_is_per_frame_and_finite(self):
        # Two 2-patch attention segments with a single head. Equal Q/K gives
        # scores within each segment and never mixes information across them.
        qkv = torch.zeros(4, 3)
        qkv[:, 0] = torch.tensor([2., 1., -2., -1.])
        qkv[:, 1] = torch.tensor([2., 1., -2., -1.])
        position = (torch.ones(4, 1), torch.zeros(4, 1))
        scores = received_attention(qkv, position, torch.tensor([0, 2, 4]), heads=1)
        self.assertEqual(scores.shape, (4,))
        self.assertTrue(torch.isfinite(scores).all())
        self.assertAlmostEqual(float(scores[:2].sum()), 1.0, places=6)
        self.assertAlmostEqual(float(scores[2:].sum()), 1.0, places=6)


if __name__ == "__main__":
    unittest.main()
