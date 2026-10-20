from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.pruning.frame_diversity import (
    embedding_suffix,
    retained_frame_count,
    select_diverse_frame_indices,
    select_random_frame_indices,
)
from scripts.pruning.random_prune import sample_seed


class FrameDiversityTests(unittest.TestCase):
    def test_embedding_suffix_reflects_actual_ratio(self) -> None:
        self.assertEqual(embedding_suffix("diverse", 0.05), "_diverse_frames_005")
        self.assertEqual(embedding_suffix("diverse", 0.20), "_diverse_frames_020")
        self.assertEqual(embedding_suffix("random", 0.20, 42), "_random_frames_020_seed42")
        self.assertEqual(embedding_suffix("none", 0.05), "")

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

    def test_random_selection_is_seeded_unique_and_temporally_sorted(self) -> None:
        first = select_random_frame_indices(20, retain_ratio=0.20, seed=42)
        second = select_random_frame_indices(20, retain_ratio=0.20, seed=42)
        other = select_random_frame_indices(20, retain_ratio=0.20, seed=43)

        self.assertEqual(first.tolist(), second.tolist())
        self.assertNotEqual(first.tolist(), other.tolist())
        self.assertEqual(len(first), 4)
        self.assertEqual(len(first.unique()), 4)
        self.assertEqual(first.tolist(), sorted(first.tolist()))

    def test_random_frame_seed_changes_by_sample_id(self) -> None:
        first_seed = sample_seed(42, "test_dia0_utt0")
        second_seed = sample_seed(42, "test_dia0_utt1")
        self.assertNotEqual(first_seed, second_seed)


if __name__ == "__main__":
    unittest.main()
