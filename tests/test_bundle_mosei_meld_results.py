"""The download bundle must not accidentally include embeddings or other runs."""

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from scripts.analysis.bundle_mosei_meld_results import RESULT_DIRS


REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "scripts" / "analysis" / "bundle_mosei_meld_results.py"


class BundleResultsTests(unittest.TestCase):
    def test_collects_only_selected_compact_outputs(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            results = root / "results"
            for name in RESULT_DIRS:
                folder = results / name
                folder.mkdir(parents=True)
                (folder / "summary.json").write_text("{}")
                (folder / "validation_curve.csv").write_text("ratio,score\n0.1,0.5\n")
                (folder / "test_metrics_curve.png").write_bytes(b"png")
                (folder / "embeddings").mkdir()
                (folder / "embeddings" / "mosei_test.pt").write_bytes(b"weights")
                (folder / "indexes").mkdir()
                (folder / "indexes" / "test.csv").write_text("sample_id\nclip\n")
            unrelated = results / "qwen2_5_vl_7b_mosei_single_label_emotion_prompt_cache_success"
            unrelated.mkdir()
            (unrelated / "summary.json").write_text("{}")
            output = root / "download"
            subprocess.run([sys.executable, str(SCRIPT), "--root", str(root),
                            "--output-dir", str(output), "--skip-attention"],
                           check=True, capture_output=True, text=True)
            manifest = json.loads((output / "bundle_manifest.json").read_text())
            self.assertEqual(manifest["result_file_count"], 3 * len(RESULT_DIRS))
            self.assertEqual(len(list((output / "results").rglob("*.csv"))), len(RESULT_DIRS))
            self.assertFalse(list(output.rglob("*.pt")))
            self.assertFalse(list(output.rglob("indexes")))
            self.assertFalse((output / "results" / unrelated.name).exists())


if __name__ == "__main__":
    unittest.main()
