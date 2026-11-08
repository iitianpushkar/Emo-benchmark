"""A failed MOSEI chunk may be skipped only after every clip was attempted."""

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from cluster.run_mosei_divprune_curve import (
    decode_failure_records, test_metrics_match, training_metrics_match,
)


class DecodeFailureRecordsTest(unittest.TestCase):
    def setUp(self):
        self.model_dir = Path("/models/qwen")
        self.protocol = {"fps": 6}
        self.payload = {
            "sample_ids": ["clip-a"],
            "embeddings": SimpleNamespace(shape=(1, 3584)),
            "metadata": [{}],
            "errors": [{"sample_id": "clip-b", "error": "TimeoutError: Decord exceeded 10s"}],
            "config": {"model_dir": str(self.model_dir), "start": 0, "limit": 2,
                       "video_protocol": self.protocol, "retain_ratio": 0.3,
                       "pruning": "divprune_projected_video_tokens", "profile_stride": 10},
        }

    def records(self):
        return decode_failure_records(self.payload, ["clip-a", "clip-b"],
                                      self.model_dir, 0, 0.3, self.protocol,
                                      10, "divprune", None, 0.5, 2)

    def test_complete_chunk_with_decode_timeout_can_continue(self):
        self.assertEqual(self.records(), self.payload["errors"])

    def test_incomplete_chunk_cannot_continue(self):
        self.payload["errors"] = []
        with self.assertRaises(ValueError):
            self.records()

    def test_model_failure_cannot_continue(self):
        self.payload["errors"][0]["error"] = "RuntimeError: CUDA out of memory"
        with self.assertRaises(ValueError):
            self.records()

    def test_wrong_clip_or_config_cannot_continue(self):
        self.payload["errors"][0]["sample_id"] = "other-clip"
        with self.assertRaises(ValueError):
            self.records()
        self.payload["errors"][0]["sample_id"] = "clip-b"
        self.payload["config"]["retain_ratio"] = 0.5
        with self.assertRaises(ValueError):
            self.records()

    def test_multilabel_emotion_prompt_requires_matching_chunk_marker(self):
        with self.assertRaises(ValueError):
            decode_failure_records(self.payload, ["clip-a", "clip-b"],
                                   self.model_dir, 0, 0.3, self.protocol,
                                   10, "divprune", None, 0.5, 2,
                                   "emotion_multilabel_task")
        self.payload["config"]["prompt_style"] = "mosei_six_binary_emotion_task_v1"
        self.assertEqual(decode_failure_records(self.payload, ["clip-a", "clip-b"],
                                                self.model_dir, 0, 0.3, self.protocol,
                                                10, "divprune", None, 0.5, 2,
                                                "emotion_multilabel_task"),
                         self.payload["errors"])

    def test_partial_training_and_test_metrics_are_accepted_only_at_matching_coverage(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            mlp = root / "mlp"
            mlp.mkdir()
            (mlp / "best_mlp.pt").touch()
            (mlp / "validation_predictions.csv").touch()
            train_coverage = {"train": {"indexed": 3, "embedded": 2, "missing": 1},
                              "valid": {"indexed": 2, "embedded": 1, "missing": 1}}
            training_report = {
                "coverage": {"train": dict(train_coverage["train"]),
                             "validation": dict(train_coverage["valid"])},
                "validation": {"sample_count": 1},
            }
            training_path = mlp / "metrics.json"
            training_path.write_text(json.dumps(training_report))
            self.assertTrue(training_metrics_match(training_path, train_coverage))

            evaluation = root / "test"
            evaluation.mkdir()
            (evaluation / "predictions.csv").touch()
            test_coverage = {"indexed": 2, "embedded": 1, "missing": 1}
            test_path = evaluation / "metrics.json"
            test_path.write_text(json.dumps({"coverage": test_coverage,
                                             "metrics": {"sample_count": 1}}))
            self.assertTrue(test_metrics_match(test_path, test_coverage))

            training_report["coverage"]["validation"]["embedded"] = 2
            training_path.write_text(json.dumps(training_report))
            self.assertFalse(training_metrics_match(training_path, train_coverage))
            self.assertFalse(test_metrics_match(test_path, {"indexed": 2, "embedded": 2,
                                                        "missing": 0}))

    def test_seven_class_metrics_require_matching_task_and_cache_subset_coverage(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            mlp = root / "mlp"
            mlp.mkdir()
            (mlp / "best_mlp.pt").touch()
            (mlp / "validation_predictions.csv").touch()
            coverage = {"train": {"indexed": 3, "embedded": 3, "missing": 0},
                        "valid": {"indexed": 2, "embedded": 2, "missing": 0},
                        "test": {"indexed": 2, "embedded": 2, "missing": 0}}
            report = {"task": "single_max_seven_class",
                      "coverage": {"train": {"indexed": 3, "embedded": 3,
                                              "missing_decoder": 0},
                                   "validation": {"indexed": 2, "embedded": 2,
                                                  "missing_decoder": 0}},
                      "validation": {"sample_count": 2}}
            metrics_path = mlp / "metrics.json"
            metrics_path.write_text(json.dumps(report))
            self.assertTrue(training_metrics_match(metrics_path, coverage, True))
            self.assertFalse(training_metrics_match(metrics_path, coverage, False))
            report["coverage"]["validation"]["embedded"] = 1
            metrics_path.write_text(json.dumps(report))
            self.assertFalse(training_metrics_match(metrics_path, coverage, True))

            evaluation = root / "test"
            evaluation.mkdir()
            (evaluation / "predictions.csv").touch()
            test_path = evaluation / "metrics.json"
            test_path.write_text(json.dumps({"task": "single_max_seven_class",
                                             "coverage": {"indexed": 2, "embedded": 2,
                                                          "missing_decoder": 0},
                                             "metrics": {"sample_count": 2}}))
            self.assertTrue(test_metrics_match(test_path, coverage["test"], True))
            self.assertFalse(test_metrics_match(test_path, coverage["test"], False))


if __name__ == "__main__":
    unittest.main()
