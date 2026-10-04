from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from frame_diversity import retained_frame_count, select_diverse_frame_indices


class FrameDiversityTests(unittest.TestCase):
    def test_twenty_percent_count(self) -> None:
        self.assertEqual(retained_frame_count(64), 13)
        self.assertEqual(retained_frame_count(4), 1)

    def test_selection_is_deterministic_unique_and_temporally_sorted(self) -> None:
        embeddings = torch.tensor(
            [
                [1.0, 0.0],
                [0.9, 0.1],
                [0.0, 1.0],
                [-1.0, 0.0],
                [0.0, -1.0],
            ]
        )
        first = select_diverse_frame_indices(embeddings, retain_ratio=0.6)
        second = select_diverse_frame_indices(embeddings, retain_ratio=0.6)
        self.assertEqual(first.tolist(), second.tolist())
        self.assertEqual(len(first), 3)
        self.assertEqual(len(first.unique()), 3)
        self.assertEqual(first.tolist(), sorted(first.tolist()))

    def test_centroid_nearest_frame_is_initial_anchor(self) -> None:
        embeddings = torch.tensor(
            [
                [1.0, 0.0],
                [0.8, 0.2],
                [0.0, 1.0],
                [-1.0, 0.0],
            ]
        )
        selected = select_diverse_frame_indices(embeddings, retain_ratio=0.25)
        self.assertEqual(selected.tolist(), [1])

    def test_invalid_embeddings_are_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "NaN or Inf"):
            select_diverse_frame_indices(torch.tensor([[1.0], [float("nan")]]))


if __name__ == "__main__":
    unittest.main()
