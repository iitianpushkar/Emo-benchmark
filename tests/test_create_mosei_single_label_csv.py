"""Check MOSEI seven-class CSV conversion without video files."""

import csv
import tempfile
import unittest
from pathlib import Path

from scripts.data.create_mosei_single_label_csv import create_csv
from scripts.data.prepare_mosei import EMOTIONS


class CreateMoseiSingleLabelCsvTests(unittest.TestCase):
    def test_unique_tied_and_all_zero(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "source.csv"
            output = Path(directory) / "single.csv"
            fields = ["sample_id", "video_id", "clip_id", "split", "text", *EMOTIONS]
            with source.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=fields)
                writer.writeheader()
                for clip, scores in enumerate(((1, 0, 0, 0, 0, 0),
                                                (1, 1, 0, 0, 0, 0),
                                                (0, 0, 0, 0, 0, 0))):
                    writer.writerow({"sample_id": f"v[{clip}]", "video_id": "v",
                                     "clip_id": clip, "split": "train", "text": " hello  world ",
                                     **dict(zip(EMOTIONS, scores))})
            counts = create_csv(source, output)
            with output.open(newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual([(row["sample_id"], row["emotion"], row["emotion_id"])
                              for row in rows], [("v[0]", "happy", "0"),
                                                 ("v[2]", "neutral", "6")])
            self.assertEqual(rows[0]["utterance"], "hello world")
            self.assertEqual(counts["train"]["tied_max"], 1)
            self.assertEqual(counts["train"]["written"], 2)


if __name__ == "__main__":
    unittest.main()
