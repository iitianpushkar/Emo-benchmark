import sys
import unittest
from pathlib import Path
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.information_horizon.information_horizon import sequence_indices, remove_visual_tokens, information_scores


class InformationHorizonTests(unittest.TestCase):
    def test_signed_scores_and_indices(self):
        mask = torch.tensor([False, True, True, False, True, False])
        self.assertEqual(sequence_indices(mask).tolist(), [0, 3, 5])
        self.assertEqual(sequence_indices(mask, 1).tolist(), [0, 2, 3, 5])
        self.assertLess(float(information_scores(torch.tensor(.1), torch.tensor(.3))), 0)
        with self.assertRaises(ValueError):
            sequence_indices(mask, 3)

    def test_real_qwen_compaction_matches_key_masking(self):
        try:
            from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import Qwen2_5_VLTextModel
            from transformers.models.qwen2_5_vl.configuration_qwen2_5_vl import Qwen2_5_VLTextConfig
        except ImportError:
            self.skipTest('Install transformers==4.57.1 for real Qwen integration test')
        torch.manual_seed(3)
        config = Qwen2_5_VLTextConfig(vocab_size=32, hidden_size=24, intermediate_size=48, num_hidden_layers=3, num_attention_heads=3, num_key_value_heads=1, rope_scaling={'rope_type': 'default', 'mrope_section': [1, 1, 2]})
        config._attn_implementation = 'eager'
        model = Qwen2_5_VLTextModel(config).eval()
        x = torch.randn(1, 7, 24)
        visual = torch.tensor([False, True, True, False, True, False, False])
        # Nontrivial original positions ensure compaction does not renumber RoPE.
        positions = torch.tensor([0, 2, 4, 5, 8, 9, 12]).view(1, 1, -1).expand(3, 1, -1)
        def forward():
            return model(inputs_embeds=x, attention_mask=torch.ones(1, 7), position_ids=positions, use_cache=False).last_hidden_state
        with torch.inference_mode():
            original = forward()
            for depth in range(3):
                for token in [None, 0, 2]:
                    keep = sequence_indices(visual, token)
                    with remove_visual_tokens(model, visual, depth, token):
                        compact = forward()
                    dropped = torch.ones(7, dtype=torch.bool)
                    dropped[keep] = False
                    def mask_keys(module, args, kwargs):
                        kwargs = dict(kwargs)
                        mask = kwargs['attention_mask'].clone()
                        mask[..., dropped] = torch.finfo(mask.dtype).min
                        kwargs['attention_mask'] = mask
                        return args, kwargs
                    handles = [l.register_forward_pre_hook(mask_keys, with_kwargs=True) for l in model.layers[depth:]]
                    try:
                        reference = forward()[:, keep]
                    finally:
                        for h in handles:
                            h.remove()
                    torch.testing.assert_close(compact, reference, atol=1e-5, rtol=1e-5)
            torch.testing.assert_close(original, forward())
            with self.assertRaisesRegex(RuntimeError, 'intentional'):
                with remove_visual_tokens(model, visual, 1):
                    raise RuntimeError('intentional')
            torch.testing.assert_close(original, forward())


if __name__ == '__main__':
    unittest.main()
