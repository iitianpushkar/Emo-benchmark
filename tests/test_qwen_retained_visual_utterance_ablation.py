from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

import torch

from cluster.qwen_sparsevlm import SparseVLMHook
from cluster.run_qwen7b_retained_visual_utterance_ablation import (
    FIELDS, RATIOS, paired_prompt, read_results, score_clip, summarize,
)


class RetainedVisualUtteranceTests(unittest.TestCase):
    def test_ratio_choices_and_prompt_pair(self):
        self.assertEqual(RATIOS["mosei"]["random"], .30)
        self.assertEqual(RATIOS["meld"]["vispruner"], .15)
        only = paired_prompt("mosei", "secret utterance", "A", False)
        both = paired_prompt("mosei", "secret utterance", "A", True)
        self.assertEqual(both, only.replace("\nAnswer", "\nUtterance: secret utterance\nAnswer"))
        self.assertNotIn("secret utterance", only)

    def test_sparse_replay_keeps_original_visual_positions_and_recycles(self):
        hook = object.__new__(SparseVLMHook)
        hook.stages = (2,)
        hook.frozen_plan = [{"target": 2, "selected_positions": [1, 3],
                             "recycled_positions": [3], "recycling_groups": {3: [2, 3]}}]
        hook._completed_stages = 0
        hook.active = torch.arange(5)
        hook.video_mask = torch.tensor([False, True, True, True, False])
        hook.final_budget = 2
        hook.full_sequence_tokens = 5
        hook.replacements = {}
        hook.stage_metadata = []
        hook.scores = hook.priority = hook.priority_rank = None
        hidden = torch.arange(5, dtype=torch.float32).reshape(1, 5, 1)
        hook._finish_stage(None, None, hidden)
        self.assertEqual(hook.next_active.tolist(), [0, 1, 3, 4])
        self.assertEqual(hook.retained_tokens, 2)
        self.assertEqual(float(hook.replacements[3]), 2.5)

    def test_paired_mosei_rows_and_resume_summary(self):
        class Inputs(dict):
            def to(self, _device):
                return self

        class Processor:
            def apply_chat_template(self, messages, **_kwargs):
                return messages[0]["content"][1]["text"]

            def __call__(self, *, text, **_kwargs):
                self.last_text = text[0]
                return Inputs(input_ids=torch.tensor([[1, 99, 99, 2]]))

        row = {"sample_id": "clip", "text": "secret", "_video_path": "/tmp/x.mp4",
               **{emotion: "1" if emotion == "happy" else "0" for emotion in
                  ("anger", "disgust", "fear", "happy", "sad", "surprise")}}
        model = SimpleNamespace(device="cpu", config=SimpleNamespace(video_token_id=99))
        cached = (None, [(torch.zeros(1), {})], {})
        calls = []

        def fake_probability(_model, inputs, _candidate_ids, _torch, method, ratio, seed,
                             frozen_sparse_plan=None):
            calls.append((method, ratio, frozen_sparse_plan))
            return ([.8, .2] if len(calls) % 2 else [.9, .1]), [1], 2, 1

        with (patch("cluster.run_qwen7b_retained_visual_utterance_ablation.identity", return_value={}),
              patch("cluster.run_qwen7b_retained_visual_utterance_ablation.load_cached", return_value=cached),
              patch("cluster.run_qwen7b_retained_visual_utterance_ablation.pruned_probability",
                    side_effect=fake_probability)):
            results = score_clip(row, "mosei", "divprune", .2, model, Processor(), Path("/tmp"),
                {"fps": 6, "min_frames": 4, "max_frames": 64, "frame_size": 224}, [0, 1], torch, 42)
        self.assertEqual(len(results), 6)
        self.assertEqual(len(calls), 12)
        happy = next(item for item in results if item["emotion"] == "happy")
        self.assertAlmostEqual(happy["ground_truth_probability_delta"], .1)
        self.assertEqual(set(happy), set(FIELDS))
        with tempfile.TemporaryDirectory() as folder:
            from cluster.run_qwen7b_retained_visual_utterance_ablation import atomic_csv
            path = Path(folder) / "results.csv"
            atomic_csv(path, results, FIELDS)
            saved, done = read_results(path, [row], "mosei", "divprune")
            self.assertEqual(done, {"clip"})
            report = summarize(saved, "mosei", "divprune", .2, 1)
            self.assertEqual(report["completed_clips"], 1)
            self.assertAlmostEqual(report["mean_ground_truth_probability_delta"], -.06666666666666665)

    def test_paired_meld_has_one_ground_truth_probability(self):
        class Inputs(dict):
            def to(self, _device):
                return self

        class Processor:
            def apply_chat_template(self, messages, **_kwargs):
                return messages[0]["content"][1]["text"]

            def __call__(self, **_kwargs):
                return Inputs(input_ids=torch.tensor([[1, 99, 2]]))

        row = {"sample_id": "clip", "utterance": "secret", "video_path": "/tmp/x.mp4",
               "emotion": "joy"}
        model = SimpleNamespace(device="cpu", config=SimpleNamespace(video_token_id=99))
        cached = (None, [(torch.zeros(1), {})], {})
        calls = 0

        def fake_probability(*_args, **_kwargs):
            nonlocal calls
            calls += 1
            probs = ([.1, .4, .1, .1, .1, .1, .1] if calls == 1 else
                     [.1, .5, .1, .1, .1, .05, .05])
            return probs, [1], 1, 1

        with (patch("cluster.run_qwen7b_retained_visual_utterance_ablation.identity", return_value={}),
              patch("cluster.run_qwen7b_retained_visual_utterance_ablation.load_cached", return_value=cached),
              patch("cluster.run_qwen7b_retained_visual_utterance_ablation.pruned_probability",
                    side_effect=fake_probability)):
            results = score_clip(row, "meld", "visionzip", .1, model, Processor(), Path("/tmp"),
                {"fps": 6, "min_frames": 4, "max_frames": 64, "frame_size": 224},
                list(range(7)), torch, 42)
        self.assertEqual(len(results), 7)
        self.assertEqual(sum(item["ground_truth_probability_delta"] != "" for item in results), 1)
        self.assertAlmostEqual(sum(item["video_only_probability"] for item in results), 1)


if __name__ == "__main__":
    unittest.main()
