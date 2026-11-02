"""Resume interrupted MOSEI extraction before revisiting decoder failures."""

import unittest
from pathlib import Path
from types import SimpleNamespace

from cluster.run_mosei_qwen2_5vl7b_baseline import (
    complete_baseline_chunk, embedding_coverage, metrics_match_coverage,
    order_chunk_work, verified_decode_errors,
)


class BaselineResumeOrderTests(unittest.TestCase):
    def test_latest_partial_checkpoint_is_first(self):
        chunks = [
            {"name": "train-errors", "state": "retry_errors", "mtime": 10, "ordinal": 0},
            {"name": "test-earlier", "state": "partial", "mtime": 20, "ordinal": 1},
            {"name": "test-latest", "state": "partial", "mtime": 30, "ordinal": 2},
            {"name": "test-new", "state": "new", "mtime": 0, "ordinal": 3},
        ]
        self.assertEqual([item["name"] for item in order_chunk_work(chunks)],
                         ["test-latest", "test-earlier", "test-new", "train-errors"])

    def test_new_run_keeps_split_order(self):
        chunks = [{"state": "new", "mtime": 0, "ordinal": i} for i in range(3)]
        self.assertEqual(order_chunk_work(chunks), chunks)

    def test_saved_failures_cover_split_without_retry(self):
        self.assertEqual(embedding_coverage(
            ["a", "c"], [{"sample_id": "b", "error": "TimeoutError: Decord exceeded 10s"}],
            ["a", "b", "c"]), {"indexed": 3, "embedded": 2, "missing": 1})
        with self.assertRaises(ValueError):
            embedding_coverage(["a", "c"], [], ["a", "b", "c"])

    def test_classifier_metrics_accept_reported_partial_coverage(self):
        import json
        import tempfile

        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder)
            for name in ("best_mlp.pt", "test_predictions.csv", "validation_predictions.csv"):
                (output / name).touch()
            split_coverage = {
                "train": {"indexed": 3, "embedded": 2, "missing": 1},
                "valid": {"indexed": 2, "embedded": 2, "missing": 0},
                "test": {"indexed": 2, "embedded": 1, "missing": 1},
            }
            report = {"coverage": {"train": split_coverage["train"],
                                   "validation": split_coverage["valid"],
                                   "test": split_coverage["test"]},
                      "validation": {"sample_count": 2}, "test": {"sample_count": 1}}
            path = output / "metrics.json"
            path.write_text(json.dumps(report))
            self.assertTrue(metrics_match_coverage(path, split_coverage))
            report["coverage"]["test"]["embedded"] = 2
            path.write_text(json.dumps(report))
            self.assertFalse(metrics_match_coverage(path, split_coverage))

    def test_successful_saved_chunk_is_reused(self):
        protocol = {"fps": 6}
        payload = {
            "sample_ids": ["a", "b"], "errors": [],
            "embeddings": SimpleNamespace(shape=(2, 3584)),
            "metadata": [{}, {}],
            "config": {"model_dir": "/models/qwen", "start": 0, "limit": 2,
                       "video_protocol": protocol, "retain_ratio": 1.0,
                       "pruning": "none", "profile_stride": 128},
        }
        self.assertTrue(complete_baseline_chunk(payload, ["a", "b"],
                                                Path("/models/qwen"), 0, protocol, 128))
        payload["errors"] = [{"sample_id": "b", "error": "TimeoutError: Decord exceeded 10s"}]
        payload["sample_ids"] = ["a"]
        self.assertFalse(complete_baseline_chunk(payload, ["a", "b"],
                                                 Path("/models/qwen"), 0, protocol, 128))
        payload["embeddings"] = SimpleNamespace(shape=(1, 3584))
        payload["metadata"] = [{}]
        self.assertEqual(verified_decode_errors(payload, ["a", "b"], Path("/models/qwen"),
                                                0, protocol, 128), payload["errors"])
        payload["errors"][0]["error"] = "RuntimeError: CUDA out of memory"
        with self.assertRaises(ValueError):
            verified_decode_errors(payload, ["a", "b"], Path("/models/qwen"),
                                   0, protocol, 128)


if __name__ == "__main__":
    unittest.main()
