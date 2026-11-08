"""Shared-decode extraction runs each pending ratio before advancing clips."""

import csv
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch

from cluster import extract_mosei_qwen2_5vl_multi as multi


class MultiRatioExtractionTest(unittest.TestCase):
    def test_one_decode_and_preprocess_per_clip_with_resume(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            index = root / "index.csv"
            with index.open("w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=("sample_id", "video_path", "utterance"))
                writer.writeheader()
                for number in range(2):
                    video = root / f"clip-{number}.mp4"
                    video.touch()
                    writer.writerow({"sample_id": f"clip-{number}", "video_path": str(video),
                                     "utterance": "hello"})
            outputs = [root / "r100.pt", root / "r050.pt"]
            argv = ["extract_mosei_qwen2_5vl_multi.py", "--index-csv", str(index),
                    "--model-dir", str(root), "--start", "0", "--limit", "2",
                    "--pruning", "divprune", "--ratio-output", "1.0", str(outputs[0]),
                    "--ratio-output", "0.5", str(outputs[1])]
            decoded, processed, forwards, prompts = [], [], [], []

            class FakeDecoder:
                def __init__(self, _timeout):
                    pass

                def decode(self, _messages, sample_id):
                    decoded.append(sample_id)
                    return None, [(torch.empty(1, 3, 224, 224), {})], {}

                def close(self):
                    pass

            class FakeProcessor:
                @classmethod
                def from_pretrained(cls, *_args, **_kwargs):
                    return cls()

                def apply_chat_template(self, messages, **_kwargs):
                    prompts.append(messages[0]["content"][1]["text"])
                    return "prompt"

                def __call__(self, *_args, **_kwargs):
                    processed.append(True)
                    return SimpleNamespace(to=lambda _device: {})

            class FakeModel:
                device = "cpu"

                @classmethod
                def from_pretrained(cls, *_args, **_kwargs):
                    return cls()

                def eval(self):
                    return self

            def fake_ratio(_model, _inputs, _args, ratio, _seed, sample_id,
                           _global_row, _common, _torch):
                forwards.append((sample_id, ratio))
                return torch.full((3584,), ratio), {"ratio": ratio}

            fake_transformers = SimpleNamespace(AutoProcessor=FakeProcessor,
                                                Qwen2_5_VLForConditionalGeneration=FakeModel)
            with (patch.dict(sys.modules, {"transformers": fake_transformers}),
                  patch.object(sys, "argv", argv),
                  patch.object(multi, "TimedVideoDecoder", FakeDecoder),
                  patch.object(multi, "extract_ratio", fake_ratio),
                  patch.object(torch.cuda, "synchronize")):
                multi.main()
                self.assertEqual(decoded, ["clip-0", "clip-1"])
                self.assertEqual(len(processed), 2)
                self.assertEqual(forwards, [("clip-0", 1.0), ("clip-0", 0.5),
                                            ("clip-1", 1.0), ("clip-1", 0.5)])
                for output in outputs:
                    payload = torch.load(output, weights_only=False)
                    self.assertEqual(payload["sample_ids"], ["clip-0", "clip-1"])

                outputs[1].unlink()
                decoded.clear()
                processed.clear()
                forwards.clear()
                multi.main()
                self.assertEqual(decoded, ["clip-0", "clip-1"])
                self.assertEqual(len(processed), 2)
                self.assertEqual(forwards, [("clip-0", 0.5), ("clip-1", 0.5)])

                prompted_outputs = [root / "emotion_r100.pt", root / "emotion_r050.pt"]
                prompted_argv = [*argv[:], "--prompt-style", "emotion_task"]
                prompted_argv[prompted_argv.index(str(outputs[0]))] = str(prompted_outputs[0])
                prompted_argv[prompted_argv.index(str(outputs[1]))] = str(prompted_outputs[1])
                with patch.object(sys, "argv", prompted_argv):
                    decoded.clear()
                    prompts.clear()
                    multi.main()
                self.assertEqual(decoded, ["clip-0", "clip-1"])
                self.assertTrue(all("Task: infer the speaker's overall emotion" in prompt
                                    for prompt in prompts))

                multilabel_outputs = [root / "multilabel_r100.pt", root / "multilabel_r050.pt"]
                multilabel_argv = [*argv[:], "--prompt-style", "emotion_multilabel_task"]
                multilabel_argv[multilabel_argv.index(str(outputs[0]))] = str(multilabel_outputs[0])
                multilabel_argv[multilabel_argv.index(str(outputs[1]))] = str(multilabel_outputs[1])
                with patch.object(sys, "argv", multilabel_argv):
                    decoded.clear()
                    prompts.clear()
                    multi.main()
                self.assertEqual(decoded, ["clip-0", "clip-1"])
                self.assertTrue(all("More than one emotion may be present" in prompt
                                    for prompt in prompts))
                for output in multilabel_outputs:
                    payload = torch.load(output, weights_only=False)
                    self.assertEqual(payload["config"]["prompt_style"],
                                     "mosei_six_binary_emotion_task_v1")

                meld_outputs = [root / "meld_r100.pt", root / "meld_r050.pt"]
                meld_argv = [*argv[:], "--prompt-style", "meld_emotion_task"]
                meld_argv[meld_argv.index(str(outputs[0]))] = str(meld_outputs[0])
                meld_argv[meld_argv.index(str(outputs[1]))] = str(meld_outputs[1])
                with patch.object(sys, "argv", meld_argv):
                    decoded.clear()
                    prompts.clear()
                    multi.main()
                self.assertEqual(decoded, ["clip-0", "clip-1"])
                self.assertTrue(all("anger, disgust, fear, joy, neutral, sadness, surprise" in prompt
                                    for prompt in prompts))
                for output in meld_outputs:
                    payload = torch.load(output, weights_only=False)
                    self.assertEqual(payload["config"]["prompt_style"],
                                     "meld_seven_class_emotion_task_v1")
                self.assertTrue(all("Utterance: hello" in prompt for prompt in prompts))
                for output in prompted_outputs:
                    payload = torch.load(output, weights_only=False)
                    self.assertEqual(payload["config"]["prompt_style"],
                                     "mosei_seven_class_emotion_task_v1")
                    self.assertEqual(payload["config"]["video_protocol"]["prompt_style"],
                                     "emotion_task")


if __name__ == "__main__":
    unittest.main()
