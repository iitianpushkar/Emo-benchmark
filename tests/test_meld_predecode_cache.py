"""MELD predecode uses the shared resumable cache and its available clip counts."""

import csv
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from cluster import predecode_mosei


class MeldPredecodeCacheTests(unittest.TestCase):
    def test_meld_indexes_parallel_workers_and_manifests(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            meld = root / "datasets" / "meld"
            for relative in ("train/train_sent_emo.csv", "dev_sent_emo.csv",
                             "test_sent_emo.csv"):
                source = meld / relative
                source.parent.mkdir(parents=True, exist_ok=True)
                source.write_text("source\n")
            counts = {"train": 2, "dev": 1, "test": 1}
            dataset_info = {"root": str(meld), "splits": {
                "train": {"rows": 2, "missing_clips": 0},
                "dev": {"rows": 2, "missing_clips": 1},
                "test": {"rows": 1, "missing_clips": 0}}}
            decoded = []

            def prepare(command, **_kwargs):
                self.assertEqual(Path(command[2]).name, "prepare_meld.py")
                index_dir = Path(command[command.index("--output-dir") + 1])
                index_dir.mkdir(parents=True, exist_ok=True)
                for split, count in counts.items():
                    with (index_dir / f"meld_{split}_index.csv").open("w", newline="") as handle:
                        writer = csv.DictWriter(handle, fieldnames=(
                            "sample_id", "video_path", "utterance", "video_exists"))
                        writer.writeheader()
                        for number in range(count):
                            writer.writerow({"sample_id": f"{split}_{number}",
                                             "video_path": str(meld / f"{split}_{number}.mp4"),
                                             "utterance": "hello", "video_exists": "True"})

            class FakeDecoders:
                def __init__(self, timeout_seconds, cache_dir):
                    self.assertions = (timeout_seconds, cache_dir)

                def decode(self, row, _protocol):
                    decoded.append(row["sample_id"])
                    return 0.01

                def close(self):
                    pass

            argv = ["predecode_meld.py", "--root", str(root), "--workers", "2",
                    "--chunk-size", "2", "--log-every", "100"]
            with (patch.object(sys, "argv", argv),
                  patch.object(predecode_mosei, "verify_meld", return_value=dataset_info),
                  patch.object(predecode_mosei, "decoder_environment", return_value={
                      "qwen-vl-utils": "test"}),
                  patch.object(predecode_mosei, "activate_live_log"),
                  patch.object(predecode_mosei.subprocess, "run", side_effect=prepare),
                  patch.object(predecode_mosei, "ParallelDecoders", FakeDecoders),
                  patch.object(predecode_mosei, "identity", return_value={}),
                  patch.object(predecode_mosei, "cache_available", return_value=False)):
                self.assertEqual(predecode_mosei.main("meld"), 0)

            output = meld / "decoded_qwen_6fps_224_v1"
            summary = json.loads((output / "summary.json").read_text())
            self.assertEqual(summary["expected"], counts)
            self.assertEqual(summary["successful"], counts)
            self.assertEqual(summary["pending"], dict.fromkeys(counts, 0))
            self.assertEqual(set(decoded), {"train_0", "train_1", "dev_0", "test_0"})
            with (output / "successful_decodes.csv").open(newline="") as handle:
                self.assertEqual(len(list(csv.DictReader(handle))), 4)


if __name__ == "__main__":
    unittest.main()
