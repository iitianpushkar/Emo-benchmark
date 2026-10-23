"""Check exact temporal and spatial coverage for uniform token retention."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.pruning.uniform_prune import build_uniform_prune_sequence_indices


class UniformPruneTests(unittest.TestCase):
    def setUp(self) -> None:
        # Two temporal groups, each with a 4 x 4 spatial grid.
        self.embeds = torch.zeros((1, 34, 4))
        self.mask = torch.tensor([[False] + [True] * 32 + [False]])
        self.positions = torch.zeros((3, 1, 34), dtype=torch.long)
        for index in range(32):
            self.positions[0, 0, index + 1] = index // 16
            self.positions[1, 0, index + 1] = (index % 16) // 4
            self.positions[2, 0, index + 1] = index % 4

    def test_exact_budget_and_each_temporal_group_covered(self) -> None:
        keep, selected = build_uniform_prune_sequence_indices(
            self.embeds, self.mask, self.positions, 0.25,
        )
        self.assertEqual(len(selected), 8)
        self.assertEqual(selected.tolist(), sorted(selected.tolist()))
        self.assertEqual(len(selected.unique()), 8)
        self.assertEqual(keep[0].item(), 0)
        self.assertEqual(keep[-1].item(), 33)
        temporal = self.positions[0, 0, selected]
        self.assertEqual((temporal == 0).sum().item(), 4)
        self.assertEqual((temporal == 1).sum().item(), 4)
        for group in (0, 1):
            group_positions = selected[temporal == group]
            height = self.positions[1, 0, group_positions]
            width = self.positions[2, 0, group_positions]
            self.assertEqual(len(height.unique()), 2)
            self.assertEqual(len(width.unique()), 2)

    def test_budget_below_group_count_and_full_retention(self) -> None:
        keep, selected = build_uniform_prune_sequence_indices(
            self.embeds, self.mask, self.positions, 1 / 32,
        )
        self.assertEqual(len(selected), 1)
        self.assertEqual(len(keep), 3)
        full_keep, full_selected = build_uniform_prune_sequence_indices(
            self.embeds, self.mask, self.positions, 1.0,
        )
        self.assertEqual(full_keep.tolist(), list(range(34)))
        self.assertEqual(full_selected.tolist(), list(range(1, 33)))

    def test_selection_is_repeatable(self) -> None:
        first = build_uniform_prune_sequence_indices(self.embeds, self.mask, self.positions, 0.3)
        second = build_uniform_prune_sequence_indices(self.embeds, self.mask, self.positions, 0.3)
        self.assertTrue(torch.equal(first[0], second[0]))
        self.assertTrue(torch.equal(first[1], second[1]))


if __name__ == "__main__":
    unittest.main()
