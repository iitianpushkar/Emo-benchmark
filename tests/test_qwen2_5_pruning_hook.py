"""Check random pruning at Qwen's projected-token boundary."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cluster.qwen2_5_divprune import DivPruneHook
from scripts.pruning.random_prune import build_random_prune_sequence_indices
from scripts.pruning.uniform_prune import build_uniform_prune_sequence_indices


class CaptureLanguageModel(torch.nn.Module):
    def forward(self, *, inputs_embeds, attention_mask, position_ids, cache_position=None):
        return inputs_embeds, attention_mask, position_ids


class QwenPruningHookTests(unittest.TestCase):
    def test_random_hook_preserves_nonvideo_tokens_and_original_positions(self) -> None:
        language_model = CaptureLanguageModel()
        model = SimpleNamespace(
            config=SimpleNamespace(video_token_id=99),
            model=SimpleNamespace(language_model=language_model),
        )
        input_ids = torch.tensor([[1, 99, 99, 99, 99, 99, 99, 2]])
        embeds = torch.arange(8 * 4, dtype=torch.float32).reshape(1, 8, 4)
        attention = torch.ones((1, 8), dtype=torch.long)
        positions = torch.arange(8).reshape(1, 1, 8).expand(3, -1, -1).clone()
        keep, selected = build_random_prune_sequence_indices(
            embeds, input_ids.eq(99), retain_ratio=0.5, seed=123,
        )

        with DivPruneHook(model, input_ids, retain_ratio=0.5,
                          method="random", random_seed=123) as hook:
            actual_embeds, actual_attention, actual_positions = language_model(
                inputs_embeds=embeds, attention_mask=attention, position_ids=positions,
            )

        self.assertEqual(hook.calls, 1)
        self.assertEqual(hook.full_tokens, 6)
        self.assertEqual(hook.retained_tokens, len(selected))
        self.assertEqual(hook.retained_sequence_tokens, len(keep))
        self.assertEqual(keep[0].item(), 0)
        self.assertEqual(keep[-1].item(), 7)
        self.assertTrue(torch.equal(actual_embeds, embeds.index_select(1, keep)))
        self.assertTrue(torch.equal(actual_attention, attention.index_select(-1, keep)))
        self.assertTrue(torch.equal(actual_positions, positions.index_select(-1, keep)))

    def test_uniform_hook_uses_temporal_positions_and_preserves_mrope(self) -> None:
        language_model = CaptureLanguageModel()
        model = SimpleNamespace(
            config=SimpleNamespace(video_token_id=99),
            model=SimpleNamespace(language_model=language_model),
        )
        input_ids = torch.tensor([[1] + [99] * 8 + [2]])
        embeds = torch.arange(10 * 4, dtype=torch.float32).reshape(1, 10, 4)
        attention = torch.ones((1, 10), dtype=torch.long)
        positions = torch.zeros((3, 1, 10), dtype=torch.long)
        for index in range(8):
            positions[0, 0, index + 1] = index // 4
            positions[1, 0, index + 1] = (index % 4) // 2
            positions[2, 0, index + 1] = index % 2
        keep, selected = build_uniform_prune_sequence_indices(
            embeds, input_ids.eq(99), positions, retain_ratio=0.5,
        )

        with DivPruneHook(model, input_ids, retain_ratio=0.5, method="uniform") as hook:
            actual_embeds, actual_attention, actual_positions = language_model(
                inputs_embeds=embeds, attention_mask=attention, position_ids=positions,
            )

        self.assertEqual(hook.calls, 1)
        self.assertEqual(hook.retained_tokens, len(selected))
        self.assertTrue(torch.equal(actual_embeds, embeds.index_select(1, keep)))
        self.assertTrue(torch.equal(actual_attention, attention.index_select(-1, keep)))
        self.assertTrue(torch.equal(actual_positions, positions.index_select(-1, keep)))


if __name__ == "__main__":
    unittest.main()
