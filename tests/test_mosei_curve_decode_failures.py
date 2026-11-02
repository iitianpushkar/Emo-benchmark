"""A failed MOSEI chunk may be skipped only after every clip was attempted."""

import unittest
from pathlib import Path
from types import SimpleNamespace

from cluster.run_mosei_divprune_curve import decode_failure_records


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


if __name__ == "__main__":
    unittest.main()
