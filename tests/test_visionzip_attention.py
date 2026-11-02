"""VisionZip received-attention reduction for unequal video segments."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cluster.qwen_vispruner_attention import QwenVisionZipCapture


class VisionZipAttentionTests(unittest.TestCase):
    def test_scores_average_heads_and_sum_queries_per_segment(self):
        # Uniform attention sums to one for every key, regardless of segment
        # length. Unequal segments catch accidental query averaging.
        attention = SimpleNamespace(num_heads=2)
        visual = SimpleNamespace(
            fullatt_block_indexes=[0],
            blocks=[SimpleNamespace(attn=attention)],
            spatial_merge_unit=4,
            get_window_index=lambda _grid: (torch.arange(3), None),
        )
        model = SimpleNamespace(model=SimpleNamespace(visual=visual))
        capture = QwenVisionZipCapture(model, torch.tensor([[3, 2, 2]]))
        capture._position_embeddings = (torch.ones(12, 2), torch.zeros(12, 2))
        capture._cu_seqlens = torch.tensor([0, 4, 12])
        capture._qkv(None, None, torch.zeros(12, 12))

        torch.testing.assert_close(
            capture.scores, torch.ones(3), rtol=0, atol=1e-7,
        )
        self.assertEqual(tuple(capture.keys.shape), (3, 2))

    def test_scores_average_heads_before_merging_patches(self):
        attention = SimpleNamespace(num_heads=2)
        visual = SimpleNamespace(
            fullatt_block_indexes=[0],
            blocks=[SimpleNamespace(attn=attention)],
            spatial_merge_unit=4,
            get_window_index=lambda _grid: (torch.arange(2), None),
        )
        model = SimpleNamespace(model=SimpleNamespace(visual=visual))
        capture = QwenVisionZipCapture(model, torch.tensor([[2, 2, 2]]))
        capture._position_embeddings = (torch.ones(8, 2), torch.zeros(8, 2))
        capture._cu_seqlens = torch.tensor([0, 8])
        qkv = torch.zeros(8, 3, 2, 2)
        qkv[:, 0, 0, 0] = 1  # Head 0 queries prefer the first four keys.
        qkv[:4, 1, 0, 0] = 2
        # Head 1 is uniform, contributing one received-attention unit per key.
        capture._qkv(None, None, qkv.reshape(8, -1))

        first_key_probability = torch.softmax(
            torch.tensor([2 / 2**0.5] * 4 + [0.0] * 4), dim=0,
        )[0]
        expected_first = (8 * first_key_probability + 1) / 2
        torch.testing.assert_close(capture.scores[0], expected_first, rtol=0, atol=1e-6)
        torch.testing.assert_close(capture.scores.sum(), torch.tensor(2.0), rtol=0, atol=1e-6)


if __name__ == "__main__":
    unittest.main()
