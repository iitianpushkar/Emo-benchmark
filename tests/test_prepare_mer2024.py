from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
import pandas as pd

from scripts.data.prepare_mer2024 import prepare_indexes


class PrepareMer2024Tests(unittest.TestCase):
    def test_builds_aligned_stratified_indexes(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            names = [f"clip_{index}" for index in range(12)]
            emotions = [emotion for emotion in ["angry", "happy", "neutral", "sad", "surprise", "worried"] for _ in range(2)]
            pd.DataFrame({"name": names, "discrete": emotions, "valence": range(12)}).to_csv(
                root / "label-disdim.csv", index=False
            )
            english = [f"en {name}" for name in names]
            english[0] = None
            chinese = [f"zh {name}" for name in names]
            chinese[1] = "&#20320;&#22909;"
            pd.DataFrame({"name": names, "chinese": chinese, "english": english}).to_csv(
                root / "label-transcription.csv", index=False
            )
            video_dir = root / "video-labeled"
            video_dir.mkdir()
            for name in names:
                (video_dir / f"{name}.mp4").touch()

            indexes = prepare_indexes(root, dev_ratio=0.5, seed=7, expected_count=12)

            self.assertEqual(len(indexes["train"]), 6)
            self.assertEqual(len(indexes["dev"]), 6)
            self.assertTrue(indexes["all"]["video_exists"].all())
            self.assertEqual(set(indexes["dev"]["emotion"]), set(emotions))
            by_id = indexes["all"].set_index("utterance_id")
            self.assertEqual(by_id.loc["clip_0", "utterance"], "zh clip_0")
            self.assertEqual(by_id.loc["clip_1", "utterance_chinese"], "\u4f60\u597d")


if __name__ == "__main__":
    unittest.main()
