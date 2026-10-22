from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.pruning.divprune import (
    build_divprune_sequence_indices,
    combined_target_retention,
    retained_token_count,
    select_diverse_token_indices,
)


class DivPruneTests(unittest.TestCase):
    def test_official_ratio_counts(self) -> None:
        self.assertEqual(retained_token_count(864, 0.098), 85)
        self.assertEqual(retained_token_count(2880, 0.098), 282)

    def test_combined_target_counts_against_unpruned_tokens(self) -> None:
        # A 20% frame sample can still contain 288 of the original 512 Qwen
        # visual tokens because the processor changes tokens per frame.
        count, ratio = combined_target_retention(512, 288, 0.15)
        self.assertEqual(count, 77)
        self.assertEqual(retained_token_count(288, ratio), 77)
        self.assertAlmostEqual(count / 512, 0.15, delta=0.001)
        self.assertEqual(combined_target_retention(100, 10, 0.15)[0], 10)

    def test_selection_is_exact_unique_sorted_and_deterministic(self) -> None:
        tokens = torch.tensor(
            [
                [1.0, 0.0],
                [0.9, 0.1],
                [0.0, 1.0],
                [-1.0, 0.0],
                [0.0, -1.0],
            ]
        )
        first = select_diverse_token_indices(tokens, retain_ratio=0.6)
        second = select_diverse_token_indices(tokens, retain_ratio=0.6)
        self.assertEqual(first.tolist(), second.tolist())
        self.assertEqual(len(first), 3)
        self.assertEqual(len(first.unique()), 3)
        self.assertEqual(first.tolist(), sorted(first.tolist()))

    def test_ratio_one_keeps_every_token(self) -> None:
        tokens = torch.randn(7, 4)
        selected = select_diverse_token_indices(tokens, retain_ratio=1.0)
        self.assertEqual(selected.tolist(), list(range(7)))

    def test_non_finite_tokens_are_rejected(self) -> None:
        tokens = torch.tensor([[1.0, 0.0], [float("nan"), 1.0]])
        with self.assertRaisesRegex(ValueError, "NaN or Inf"):
            select_diverse_token_indices(tokens)

    def test_sequence_pruning_keeps_all_non_visual_positions(self) -> None:
        inputs = torch.randn(1, 10, 4)
        video_mask = torch.tensor(
            [[False, False, True, True, True, True, True, True, False, False]]
        )
        keep, selected_visual = build_divprune_sequence_indices(
            inputs,
            video_mask,
            retain_ratio=0.5,
        )
        self.assertEqual(len(selected_visual), 3)
        self.assertTrue({0, 1, 8, 9}.issubset(set(keep.tolist())))
        self.assertEqual(len(keep), 7)
        self.assertEqual(keep.tolist(), sorted(keep.tolist()))


if __name__ == "__main__":
    unittest.main()
