from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import pandas as pd

from scripts.data.prepare_meld import prepare_split


class PrepareMeldTests(unittest.TestCase):
    def test_indexes_available_videos_and_preserves_missing_rows_when_requested(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            metadata_dir = root / "train"
            video_dir = metadata_dir / "train_splits"
            video_dir.mkdir(parents=True)
            (video_dir / "dia1_utt0.mp4").touch()
            pd.DataFrame(
                {
                    "Utterance": ["Hi", "Bye"],
                    "Emotion": ["Joy", "Neutral"],
                    "Dialogue_ID": [1, 1],
                    "Utterance_ID": [0, 1],
                }
            ).to_csv(metadata_dir / "train_sent_emo.csv", index=False)

            available = prepare_split(root, "train", include_missing=False, limit=None)
            self.assertEqual(available.sample_id.tolist(), ["train_dia1_utt0"])
            self.assertEqual(available.video_exists.tolist(), [True])
            self.assertEqual(available.video_path.tolist(), [str((video_dir / "dia1_utt0.mp4").resolve())])

            all_rows = prepare_split(root, "train", include_missing=True, limit=None)
            self.assertEqual(all_rows.sample_id.tolist(), ["train_dia1_utt0", "train_dia1_utt1"])
            self.assertEqual(all_rows.video_exists.tolist(), [True, False])


if __name__ == "__main__":
    unittest.main()
