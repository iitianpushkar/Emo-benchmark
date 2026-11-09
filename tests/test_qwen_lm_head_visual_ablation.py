from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

from cluster.qwen_no_visual_tokens import NoVisualTokensHook
from cluster.run_qwen7b_lm_head_visual_ablation import (MOSEI_EMOTIONS, RESULT_FIELDS,
    letter_map, probability_pair, prompt_for, read_results, summarize,
    score_clip, utterance_only_probability, video_only_probability)


class LanguageModel(torch.nn.Module):
    def forward(self, *, inputs_embeds, attention_mask, position_ids,
                use_cache=False, cache_position=None):
        return inputs_embeds, attention_mask, position_ids, cache_position


def test_no_visual_hook_removes_only_video_embeddings_and_positions():
    lm = LanguageModel()
    model = SimpleNamespace(config=SimpleNamespace(video_token_id=99),
                            model=SimpleNamespace(language_model=lm))
    ids = torch.tensor([[1, 99, 99, 2, 99, 3]])
    embeddings = torch.arange(12).reshape(1, 6, 2).float()
    mask = torch.ones(1, 6)
    positions = torch.arange(18).reshape(3, 1, 6)
    with NoVisualTokensHook(model, ids) as hook:
        out = lm(inputs_embeds=embeddings, attention_mask=mask,
                 position_ids=positions, use_cache=False,
                 cache_position=torch.arange(6))
    assert hook.full_visual_tokens == 3
    assert out[0].tolist() == embeddings[:, [0, 3, 5]].tolist()
    assert out[1].shape == (1, 3)
    assert out[2].tolist() == positions[:, :, [0, 3, 5]].tolist()
    assert out[3].tolist() == [0, 3, 5]
    assert len(lm._forward_pre_hooks) == 0


def test_mosei_is_multilabel_and_summary_handles_no_positive_in_subset(tmp_path):
    assert set(letter_map("mosei").values()) == set(MOSEI_EMOTIONS)
    assert "emotion A=anger" in prompt_for("mosei", "hello", "A")
    assert "utterance alone" in prompt_for("mosei", "hello", "A", "utterance_only")
    assert "hello" not in prompt_for("mosei", "hello", "A", "video_only")
    assert "Utterance:" not in prompt_for("mosei", "hello", "A", "video_only")
    rows = []
    for code, emotion in letter_map("mosei").items():
        row = dict.fromkeys(RESULT_FIELDS, "")
        row.update(sample_id="clip", dataset="mosei", emotion_code=code, emotion=emotion,
                   ground_truth=0, full_probability=0.2, no_visual_probability=0.3,
                   utterance_only_probability=0.4,
                   video_only_probability=0.1,
                   delta_full_minus_no_visual=-0.1,
                   delta_full_minus_utterance_only=-0.2,
                   delta_no_visual_minus_utterance_only=-0.1,
                   delta_full_minus_video_only=0.1,
                   delta_no_visual_minus_video_only=0.2,
                   delta_video_only_minus_utterance_only=-0.3,
                   full_ground_truth_probability=0.8,
                   no_visual_ground_truth_probability=0.7,
                   utterance_only_ground_truth_probability=0.6,
                   video_only_ground_truth_probability=0.9,
                   ground_truth_probability_delta=0.1,
                   ground_truth_probability_delta_full_minus_utterance_only=0.2,
                   ground_truth_probability_delta_no_visual_minus_utterance_only=0.1,
                   ground_truth_probability_delta_full_minus_video_only=-0.1,
                   ground_truth_probability_delta_no_visual_minus_video_only=-0.2,
                   ground_truth_probability_delta_video_only_minus_utterance_only=0.3,
                   full_prediction=0, no_visual_prediction=0,
                   utterance_only_prediction=0, video_only_prediction=0,
                   visual_tokens=3)
        rows.append(row)
    path = tmp_path / "results.csv"
    import csv
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=RESULT_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    saved, done = read_results(path, [{"sample_id": "clip"}], "mosei")
    assert len(saved) == 6 and done == {"clip"}
    report = summarize(saved, "mosei", 1)
    assert report["completed_clips"] == 1
    assert report["mean_positive_ground_truth_probability_change"] is None
    assert report["mean_positive_ground_truth_probability_change_full_minus_utterance_only"] is None
    assert report["mean_positive_ground_truth_probability_change_full_minus_video_only"] is None


def test_no_visual_hook_requires_video_tokens():
    model = SimpleNamespace(config=SimpleNamespace(video_token_id=99),
                            model=SimpleNamespace(language_model=LanguageModel()))
    try:
        NoVisualTokensHook(model, torch.tensor([[1, 2, 3]]))
    except ValueError as exc:
        assert "no expanded video tokens" in str(exc)
    else:
        raise AssertionError("Expected a missing-video-token failure")


def test_probability_pair_uses_same_prompt_and_changes_only_video_tokens():
    class TestLanguageModel(torch.nn.Module):
        def forward(self, *, inputs_embeds, attention_mask, position_ids, use_cache=False):
            values = inputs_embeds.sum(dim=1, keepdim=True)
            return values

    class TestVisionModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.language_model = TestLanguageModel()

        def forward(self, *, input_ids, attention_mask, position_ids,
                    use_cache=False, return_dict=True):
            embeds = torch.stack((input_ids.float(), -input_ids.float()), dim=-1)
            hidden = self.language_model(inputs_embeds=embeds,
                attention_mask=attention_mask, position_ids=position_ids, use_cache=use_cache)
            return SimpleNamespace(last_hidden_state=hidden)

    model = SimpleNamespace(config=SimpleNamespace(video_token_id=99),
                            model=TestVisionModel(), lm_head=torch.nn.Identity())
    inputs = {"input_ids": torch.tensor([[1, 99, 2]]),
              "attention_mask": torch.ones(1, 3),
              "position_ids": torch.arange(3).reshape(1, 1, 3)}
    full, absent, count = probability_pair(model, inputs, [0, 1], torch)
    assert count == 1
    assert full[0] > absent[0] > 0.5
    assert abs(sum(full) - 1) < 1e-6
    assert abs(sum(absent) - 1) < 1e-6


def test_utterance_only_never_passes_visual_inputs():
    class Inputs(dict):
        def to(self, device):
            assert device == "cpu"
            return self

    class Processor:
        def apply_chat_template(self, messages, **kwargs):
            assert all(part["type"] == "text" for part in messages[0]["content"])
            assert "utterance alone" in messages[0]["content"][0]["text"]
            return "prompt"

        def __call__(self, **kwargs):
            assert set(kwargs) == {"text", "return_tensors"}
            return Inputs(input_ids=torch.tensor([[1, 2]]),
                          attention_mask=torch.ones(1, 2),
                          position_ids=torch.arange(2).reshape(1, 1, 2))

    class TextModel:
        def __call__(self, **kwargs):
            assert not any(key.startswith("pixel") or key.startswith("video") for key in kwargs)
            return SimpleNamespace(last_hidden_state=torch.tensor([[[1.0, 0.0]]]))

    model = SimpleNamespace(config=SimpleNamespace(video_token_id=99),
                            model=TextModel(), lm_head=torch.nn.Identity(), device="cpu")
    probabilities = utterance_only_probability(
        model, Processor(), "mosei", "hello", "A", [0, 1], torch)
    assert probabilities[0] > probabilities[1]


def test_video_only_uses_frames_without_utterance():
    class Inputs(dict):
        def to(self, device):
            assert device == "cpu"
            return self

    class Processor:
        def apply_chat_template(self, messages, **kwargs):
            content = messages[0]["content"]
            assert content[0]["type"] == "video"
            assert "the video alone" in content[1]["text"]
            assert "Utterance:" not in content[1]["text"]
            assert "secret transcript" not in content[1]["text"]
            return "video-only prompt"

        def __call__(self, **kwargs):
            assert "videos" in kwargs and "images" in kwargs
            return Inputs(input_ids=torch.tensor([[1, 99, 2]]),
                          attention_mask=torch.ones(1, 3),
                          position_ids=torch.arange(3).reshape(1, 1, 3))

    class VideoModel:
        def __call__(self, **kwargs):
            return SimpleNamespace(last_hidden_state=torch.tensor([[[1.0, 0.0]]]))

    model = SimpleNamespace(config=SimpleNamespace(video_token_id=99),
                            model=VideoModel(), lm_head=torch.nn.Identity(), device="cpu")
    from pathlib import Path
    probabilities = video_only_probability(
        model, Processor(), "mosei", "A", Path("/tmp/video.mp4"),
        {"fps": 6, "min_frames": 4, "max_frames": 64, "frame_size": 224},
        None, [torch.zeros(4, 3, 224, 224)], [{}], {}, [0, 1], torch)
    assert probabilities[0] > probabilities[1]


def test_score_clip_writes_all_four_conditions_for_both_datasets():
    class Inputs(dict):
        def to(self, _device):
            return self

    class Processor:
        def apply_chat_template(self, messages, **kwargs):
            return "prompt"

        def __call__(self, **kwargs):
            return Inputs(input_ids=torch.tensor([[1, 99, 2]]))

    model = SimpleNamespace(device="cpu")
    cached = (None, [(torch.zeros(4, 3, 224, 224), {})], {})
    for dataset in ("mosei", "meld"):
        row = {"sample_id": "clip", "utterance": "hello", "_video_path": "/tmp/x.mp4",
               "video_path": "/tmp/x.mp4", "emotion": "joy",
               **{name: "1" if name == "happy" else "0" for name in MOSEI_EMOTIONS}}
        count = 6 if dataset == "mosei" else 7
        full = ([0.8, 0.2] if dataset == "mosei" else [1 / 7] * 7)
        no_visual = ([0.6, 0.4] if dataset == "mosei" else [1 / 7] * 7)
        text = ([0.3, 0.7] if dataset == "mosei" else [1 / 7] * 7)
        visual = ([0.7, 0.3] if dataset == "mosei" else [1 / 7] * 7)
        with (patch("cluster.run_qwen7b_lm_head_visual_ablation.identity", return_value={}),
              patch("cluster.run_qwen7b_lm_head_visual_ablation.load_cached", return_value=cached),
              patch("cluster.run_qwen7b_lm_head_visual_ablation.probability_pair",
                    return_value=(full, no_visual, 3)),
              patch("cluster.run_qwen7b_lm_head_visual_ablation.utterance_only_probability",
                    return_value=text),
              patch("cluster.run_qwen7b_lm_head_visual_ablation.video_only_probability",
                    return_value=visual)):
            result = score_clip(row, dataset, model, Processor(), None,
                {"fps": 6, "min_frames": 4, "max_frames": 64, "frame_size": 224},
                [0, 1], torch)
        assert len(result) == count
        assert all(set(item) == set(RESULT_FIELDS) for item in result)
        assert all(item["video_only_probability"] != "" for item in result)
        if dataset == "mosei":
            happy = next(item for item in result if item["emotion"] == "happy")
            assert happy["ground_truth"] == 1
            assert abs(happy["ground_truth_probability_delta_full_minus_video_only"] - 0.1) < 1e-6


class VisualAblationTests(unittest.TestCase):
    def test_hook(self):
        test_no_visual_hook_removes_only_video_embeddings_and_positions()

    def test_multilabel_summary(self):
        import tempfile
        from pathlib import Path
        with tempfile.TemporaryDirectory() as folder:
            test_mosei_is_multilabel_and_summary_handles_no_positive_in_subset(Path(folder))

    def test_missing_video(self):
        test_no_visual_hook_requires_video_tokens()

    def test_probability_pair(self):
        test_probability_pair_uses_same_prompt_and_changes_only_video_tokens()

    def test_utterance_only(self):
        test_utterance_only_never_passes_visual_inputs()

    def test_video_only(self):
        test_video_only_uses_frames_without_utterance()

    def test_clip_output(self):
        test_score_clip_writes_all_four_conditions_for_both_datasets()
