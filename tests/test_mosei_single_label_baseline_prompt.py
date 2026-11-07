"""Check seven-class baseline routing without loading Qwen."""

import csv
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from cluster.mosei_prompt import make_mosei_prompt
from cluster.run_mosei_qwen2_5vl7b_baseline import (
    complete_baseline_chunk, metrics_match_coverage, single_label_counts,
)
from scripts.data.prepare_mosei import EMOTIONS


class SingleLabelBaselinePromptTests(unittest.TestCase):
    def test_task_prompt_has_all_classes_without_true_label(self):
        prompt = make_mosei_prompt("A sample utterance", "emotion_task")
        self.assertIn("happy, sad, anger, surprise, disgust, fear, neutral", prompt)
        self.assertIn("Utterance: A sample utterance", prompt)
        self.assertEqual(make_mosei_prompt("A sample utterance", "utterance_only"),
                         "A sample utterance")
        multi = make_mosei_prompt("A sample utterance", "emotion_multilabel_task")
        self.assertIn("More than one emotion may be present", multi)
        self.assertNotIn("exactly one", multi)

    def test_prompted_chunk_cannot_reuse_old_prompt(self):
        protocol = {"fps": 6}
        payload = {"sample_ids": ["a"], "errors": [], "embeddings": SimpleNamespace(shape=(1, 3584)),
                   "metadata": [{}], "config": {"model_dir": "/models/qwen", "start": 0,
                   "limit": 1, "video_protocol": protocol, "retain_ratio": 1.0,
                   "pruning": "none", "profile_stride": 128}}
        self.assertFalse(complete_baseline_chunk(payload, ["a"], Path("/models/qwen"),
                                                 0, protocol, 128, "emotion_task"))
        payload["config"]["prompt_style"] = "mosei_seven_class_emotion_task_v1"
        self.assertTrue(complete_baseline_chunk(payload, ["a"], Path("/models/qwen"),
                                                0, protocol, 128, "emotion_task"))

    def test_filtered_csv_counts_and_classifier_coverage(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            source = root / "single.csv"
            fields = ["sample_id", "split", "emotion", "emotion_id", *EMOTIONS]
            with source.open("w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=fields)
                writer.writeheader()
                for split, label, label_id in (("train", "happy", 0),
                                               ("valid", "sad", 1),
                                               ("test", "neutral", 6)):
                    scores = dict.fromkeys(EMOTIONS, 0)
                    if label != "neutral":
                        scores[label] = 1
                    writer.writerow({"sample_id": f"v_{split}[0]", "split": split,
                                     "emotion": label, "emotion_id": label_id, **scores})
            self.assertEqual(single_label_counts(source), {"train": 1, "valid": 1, "test": 1})
            outputs = root / "outputs"
            outputs.mkdir()
            for name in ("best_mlp.pt", "test_predictions.csv", "validation_predictions.csv"):
                (outputs / name).touch()
            coverage = {split: {"indexed": 1, "embedded": 1, "missing": 0}
                        for split in ("train", "valid", "test")}
            report = {"coverage": {"train": {"indexed": 1, "embedded": 1, "missing_decoder": 0},
                                   "validation": {"indexed": 1, "embedded": 1, "missing_decoder": 0},
                                   "test": {"indexed": 1, "embedded": 1, "missing_decoder": 0}},
                      "validation": {"sample_count": 1}, "test": {"sample_count": 1}}
            path = outputs / "metrics.json"
            path.write_text(json.dumps(report))
            self.assertTrue(metrics_match_coverage(path, coverage, True))


if __name__ == "__main__":
    unittest.main()
