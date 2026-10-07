from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from random_prune import (
    build_random_prune_sequence_indices,
    sample_seed,
    select_random_token_indices,
)


class RandomPruneTests(unittest.TestCase):
    def test_selection_is_exact_unique_sorted_and_reproducible(self) -> None:
        first = select_random_token_indices(20, retain_ratio=0.30, seed=42)
        second = select_random_token_indices(20, retain_ratio=0.30, seed=42)
        self.assertEqual(first.tolist(), second.tolist())
        self.assertEqual(len(first), 6)
        self.assertEqual(len(first.unique()), 3)
        self.assertEqual(first.tolist(), sorted(first.tolist()))

    def test_different_seeds_change_selection(self) -> None:
        first = select_random_token_indices(100, retain_ratio=0.30, seed=1)
        second = select_random_token_indices(100, retain_ratio=0.30, seed=2)
        self.assertNotEqual(first.tolist(), second.tolist())

    def test_ratio_one_keeps_every_token(self) -> None:
        selected = select_random_token_indices(7, retain_ratio=1.0, seed=42)
        self.assertEqual(selected.tolist(), list(range(7)))

    def test_sample_seed_is_stable_and_sample_specific(self) -> None:
        self.assertEqual(sample_seed(42, "sample-a"), sample_seed(42, "sample-a"))
        self.assertNotEqual(sample_seed(42, "sample-a"), sample_seed(42, "sample-b"))

    def test_sequence_pruning_keeps_all_non_visual_positions(self) -> None:
        inputs = torch.randn(1, 10, 4)
        video_mask = torch.tensor(
            [[False, False, True, True, True, True, True, True, False, False]]
        )
        keep, selected_visual = build_random_prune_sequence_indices(
            inputs,
            video_mask,
            retain_ratio=0.5,
            seed=42,
        )
        self.assertEqual(len(selected_visual), 3)
        self.assertTrue({0, 1, 8, 9}.issubset(set(keep.tolist())))
        self.assertEqual(len(keep), 7)
        self.assertEqual(keep.tolist(), sorted(keep.tolist()))


if __name__ == "__main__":
    unittest.main()
