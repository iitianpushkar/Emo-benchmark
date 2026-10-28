from __future__ import annotations

import csv
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


class PrepareMoseiKaggleTests(unittest.TestCase):
    def test_separate_csv_and_nested_raw_directory(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            csv_path = root / "mosei_aligned.csv"
            raw_dir = root / "Raw" / "Raw"
            clip = raw_dir / "video123" / "7.mp4"
            clip.parent.mkdir(parents=True)
            clip.touch()
            (clip.parent / "8.mp4").touch()
            (clip.parent / "9.mp4").touch()
            columns = (
                "sample_id", "video_id", "clip_id", "split", "text",
                "start_seconds", "end_seconds", "happy", "sad", "anger",
                "surprise", "disgust", "fear",
            )
            with csv_path.open("w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=columns)
                writer.writeheader()
                writer.writerow({
                    "sample_id": "video123[7]", "video_id": "video123", "clip_id": "7",
                    "split": "valid", "text": "sample transcript", "start_seconds": "1",
                    "end_seconds": "2", "happy": "1", "sad": "0", "anger": "0",
                    "surprise": "0.5", "disgust": "0", "fear": "0",
                })
                writer.writerow({
                    "sample_id": "video123[8]", "video_id": "video123", "clip_id": "8",
                    "split": "valid", "text": "tie", "start_seconds": "2",
                    "end_seconds": "3", "happy": "1", "sad": "1", "anger": "0",
                    "surprise": "0", "disgust": "0", "fear": "0",
                })
                writer.writerow({
                    "sample_id": "video123[9]", "video_id": "video123", "clip_id": "9",
                    "split": "valid", "text": "neutral", "start_seconds": "3",
                    "end_seconds": "4", "happy": "0", "sad": "0", "anger": "0",
                    "surprise": "0", "disgust": "0", "fear": "0",
                })
            output = root / "indexes"
            script = Path(__file__).resolve().parents[1] / "scripts/data/prepare_mosei.py"
            subprocess.run([
                sys.executable, str(script), "--aligned-csv", str(csv_path),
                "--raw-dir", str(raw_dir), "--output-dir", str(output),
                "--target-mode", "single_max", "--tie-policy", "exclude",
                "--zero-policy", "neutral",
            ], check=True, capture_output=True, text=True)
            with (output / "mosei_valid_index.csv").open(newline="") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), 2)
            self.assertEqual(Path(rows[0]["video_path"]).resolve(), clip.resolve())
            self.assertEqual((rows[0]["emotion"], rows[0]["emotion_id"], rows[0]["label_status"]),
                             ("happy", "0", "unique_max"))
            self.assertEqual((rows[1]["emotion"], rows[1]["emotion_id"], rows[1]["label_status"]),
                             ("neutral", "6", "all_zero"))
            self.assertEqual(rows[0]["happy_label"], "1")
            self.assertEqual(rows[0]["surprise_label"], "1")
            self.assertEqual(rows[0]["sad_label"], "0")

            multilabel_output = root / "multilabel_indexes"
            subprocess.run([
                sys.executable, str(script), "--aligned-csv", str(csv_path),
                "--raw-dir", str(raw_dir), "--output-dir", str(multilabel_output),
            ], check=True, capture_output=True, text=True)
            with (multilabel_output / "mosei_valid_index.csv").open(newline="") as handle:
                original_rows = list(csv.DictReader(handle))
            self.assertEqual(len(original_rows), 3)
            self.assertEqual(original_rows[1]["happy_label"], "1")
            self.assertEqual(original_rows[1]["sad_label"], "1")
            self.assertEqual(original_rows[1]["emotion"], "multilabel_placeholder")


if __name__ == "__main__":
    unittest.main()
