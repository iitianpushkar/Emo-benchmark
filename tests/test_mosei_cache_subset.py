"""Cache success manifest and seven-class label intersection."""

import csv
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from cluster.mosei_cache_subset import cached_single_label_rows, resolve_cache_frames, write_subset
from cluster.mosei_decoded_cache import cache_paths, identity


class CacheSubsetTests(unittest.TestCase):
    def test_only_valid_cached_single_label_clips_are_selected(self):
        with tempfile.TemporaryDirectory() as folder, patch(
            "cluster.mosei_decoded_cache.qwen_utils_version", return_value="test"
        ):
            root = Path(folder)
            cache_dir, raw_dir = root / "cache" / "frames", root / "Raw"
            source = root / "single.csv"
            protocol = {"fps": 6, "min_frames": 4, "max_frames": 64, "frame_size": 224}
            rows = [
                ("train", "train_ok", "happy", "unique_max", True),
                ("train", "train_failed", "sad", "unique_max", False),
                ("valid", "valid_ok", "neutral", "all_zero", True),
                ("test", "test_ok", "anger", "unique_max", True),
                ("test", "test_stale", "fear", "unique_max", True),
            ]
            manifest_rows = []
            with source.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=[
                    "sample_id", "split", "video_id", "clip_id", "emotion", "emotion_id", "label_status"
                ])
                writer.writeheader()
                for split, video_id, emotion, status, successful in rows:
                    sample_id = f"{video_id}[0]"
                    writer.writerow({"sample_id": sample_id, "split": split,
                                     "video_id": video_id, "clip_id": "0",
                                     "emotion": emotion, "emotion_id": 0, "label_status": status})
                    path = raw_dir / video_id / "0.mp4"
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_bytes(b"video")
                    if successful:
                        data_path, metadata_path = cache_paths(cache_dir, sample_id)
                        data_path.parent.mkdir(parents=True, exist_ok=True)
                        data_path.write_bytes(b"cached")
                        if video_id != "test_stale":
                            metadata_path.write_text(json.dumps(identity(sample_id, path, protocol)))
                        manifest_rows.append({"split": split, "sample_id": sample_id,
                                              "cache_path": str(data_path)})
            manifest = cache_dir.parent / "successful_decodes.csv"
            with manifest.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=["split", "sample_id", "cache_path"])
                writer.writeheader()
                writer.writerows(manifest_rows)

            self.assertEqual(resolve_cache_frames(cache_dir.parent), cache_dir.resolve())
            self.assertEqual(resolve_cache_frames(cache_dir), cache_dir.resolve())

            fields, selected, summary = cached_single_label_rows(source, cache_dir, raw_dir, protocol)
            self.assertEqual(summary["selected"], {"train": 1, "valid": 1, "test": 1})
            self.assertEqual(summary["stale_success_entries"], 1)
            self.assertEqual([row["sample_id"] for row in selected],
                             ["train_ok[0]", "valid_ok[0]", "test_ok[0]"])
            output = root / "subset.csv"
            write_subset(output, fields, selected)
            with output.open(newline="", encoding="utf-8") as handle:
                self.assertEqual(len(list(csv.DictReader(handle))), 3)


if __name__ == "__main__":
    unittest.main()
