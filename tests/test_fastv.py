"""FastV's attention ranking and Qwen decoder-sequence slicing."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cluster.qwen_fastv import last_query_attention, select_fastv_indices, slice_layer_inputs


class FastVTests(unittest.TestCase):
    def test_selects_top_video_attention_and_keeps_text(self):
        scores = torch.tensor([0., 0.2, 0.8, 0.1, 0.7, 0.4, 0.])
        video = torch.tensor([False, True, True, True, True, True, False])
        keep, selected = select_fastv_indices(scores, video, 0.4)
        self.assertEqual(selected.tolist(), [2, 4])
        self.assertEqual(keep.tolist(), [0, 2, 4, 6])

    def test_deterministic_tie_and_full_retention(self):
        scores = torch.ones(5)
        video = torch.tensor([False, True, True, True, False])
        self.assertEqual(select_fastv_indices(scores, video, 1 / 3)[1].tolist(), [1])
        self.assertEqual(select_fastv_indices(scores, video, 1.0)[0].tolist(), list(range(5)))

    def test_slices_hidden_mask_rope_and_cache_position(self):
        hidden = torch.arange(7 * 3).reshape(1, 7, 3)
        mask = torch.triu(torch.ones(7, 7), diagonal=1).reshape(1, 1, 7, 7)
        cos = torch.arange(3 * 7 * 2).reshape(3, 1, 7, 2)
        sin = cos + 1
        kwargs = {
            "attention_mask": mask,
            "position_embeddings": (cos, sin),
            "position_ids": torch.arange(7).reshape(1, 7),
            "cache_position": torch.arange(7),
            "use_cache": False,
        }
        keep = torch.tensor([0, 2, 4, 6])
        args, updated = slice_layer_inputs((hidden,), kwargs, keep, prune_hidden=True)
        self.assertTrue(torch.equal(args[0], hidden[:, keep]))
        self.assertTrue(torch.equal(updated["attention_mask"], mask[:, :, keep][:, :, :, keep]))
        self.assertTrue(torch.equal(updated["position_embeddings"][0], cos[:, :, keep]))
        self.assertTrue(torch.equal(updated["position_embeddings"][1], sin[:, :, keep]))
        self.assertEqual(updated["position_ids"].tolist(), [[0, 2, 4, 6]])
        self.assertEqual(updated["cache_position"].tolist(), [0, 2, 4, 6])
        later, _ = slice_layer_inputs(args, kwargs, keep, prune_hidden=False)
        self.assertTrue(torch.equal(later[0], args[0]))

    def test_last_query_attention_uses_qwen_rope_and_gqa(self):
        q = torch.zeros((1, 4, 24))
        k = torch.zeros((1, 4, 12))
        q[0, -1, 0] = 2.0
        q[0, -1, 12] = 2.0
        k[0, 1, 0] = 1.0
        k[0, 2, 0] = 3.0
        cosine = torch.ones((3, 1, 4, 12))
        sine = torch.zeros_like(cosine)
        mask = torch.triu(torch.full((4, 4), -1e4), diagonal=1).reshape(1, 1, 4, 4)
        scores = last_query_attention(q, k, (cosine, sine), [2, 2, 2], 2, 1, mask)
        self.assertEqual(scores.shape, (4,))
        self.assertEqual(int(scores.argmax()), 2)
        self.assertAlmostEqual(float(scores.sum()), 1.0, places=6)


if __name__ == "__main__":
    unittest.main()
