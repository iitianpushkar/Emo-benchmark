"""Cluster launchers resume across operational settings with stable chunks."""

import json
import tempfile
import unittest
from pathlib import Path

from cluster.resume_config import record_resume_config


class ResumeConfigTests(unittest.TestCase):
    def test_changed_gpu_and_revision_are_recorded_without_blocking(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "run_config.json"
            first = {"chunk_size": 500, "code_revision": "old", "gpu": "20 GB"}
            second = {"chunk_size": 500, "code_revision": "new", "gpu": "40 GB"}
            record_resume_config(path, first)
            record_resume_config(path, second)
            self.assertEqual(json.loads(path.read_text()), second)
            self.assertEqual(json.loads(path.with_name("resume_configs.json").read_text()),
                             [first, second])

    def test_changed_chunk_size_is_blocked_without_overwriting_history(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "run_config.json"
            first = {"chunk_size": 500, "code_revision": "old"}
            record_resume_config(path, first)
            with self.assertRaisesRegex(ValueError, "saved chunk_size=500"):
                record_resume_config(path, {"chunk_size": 250, "code_revision": "new"})
            self.assertEqual(json.loads(path.read_text()), first)
            self.assertFalse(path.with_name("resume_configs.json").exists())

    def test_legacy_face_config_uses_historic_default_chunk_size(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "run_config.json"
            path.write_text('{"code_revision": "old"}\n')
            record_resume_config(path, {"chunk_size": 500, "code_revision": "new"},
                                 legacy_chunk_size=500)
            self.assertEqual(json.loads(path.read_text())["chunk_size"], 500)


if __name__ == "__main__":
    unittest.main()
