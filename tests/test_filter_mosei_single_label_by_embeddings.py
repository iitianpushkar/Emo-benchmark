"""Verify that the reusable MOSEI subset joins by ID and original split."""

import csv
import tempfile
import unittest
from pathlib import Path

from scripts.data.filter_mosei_single_label_by_embeddings import filter_rows


class FilterMoseiSingleLabelTests(unittest.TestCase):
    def test_intersection_preserves_source_rows(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            source, ids, output = (root / name for name in ("single.csv", "ids.csv", "out.csv"))
            with source.open("w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=[
                    "sample_id", "split", "emotion", "emotion_id", "label_status", "utterance"])
                writer.writeheader()
                writer.writerows([
                    {"sample_id": "v[0]", "split": "train", "emotion": "happy",
                     "emotion_id": "0", "label_status": "unique_max", "utterance": "a"},
                    {"sample_id": "v[1]", "split": "train", "emotion": "neutral",
                     "emotion_id": "6", "label_status": "all_zero", "utterance": "b"},
                    {"sample_id": "w[0]", "split": "test", "emotion": "sad",
                     "emotion_id": "1", "label_status": "unique_max", "utterance": "c"},
                ])
            with ids.open("w", newline="") as handle:
                writer = csv.writer(handle)
                writer.writerows([("split", "sample_id"), ("train", "v[1]"),
                                  ("test", "w[0]"), ("test", "tied[0]")])
            report = filter_rows(source, ids, output)
            with output.open(newline="") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual([row["sample_id"] for row in rows], ["v[1]", "w[0]"])
            self.assertEqual(rows[0]["emotion"], "neutral")
            self.assertEqual(report["written_by_split"], {"train": 1, "valid": 0, "test": 1})
            self.assertEqual(report["successful_ids_without_eligible_label"], 1)


if __name__ == "__main__":
    unittest.main()
