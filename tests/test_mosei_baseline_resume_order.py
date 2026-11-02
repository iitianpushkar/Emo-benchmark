"""Resume interrupted MOSEI extraction before revisiting decoder failures."""

import unittest
from pathlib import Path
from types import SimpleNamespace

from cluster.run_mosei_qwen2_5vl7b_baseline import (
    complete_baseline_chunk, order_chunk_work,
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


if __name__ == "__main__":
    unittest.main()
