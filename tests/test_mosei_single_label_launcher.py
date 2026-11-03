"""Validate seven-class evaluation routing without loading Qwen or PyTorch."""

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from cluster.evaluate_mosei_single_label import summarize_reports
from scripts.data.prepare_mosei import EMOTIONS, SINGLE_LABELS, single_label

ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = ROOT / "cluster" / "evaluate_mosei_single_label.py"


class SingleLabelLauncherTests(unittest.TestCase):
    def test_tie_exclusion_and_neutral_mapping(self):
        scores = dict.fromkeys(EMOTIONS, 0.0)
        self.assertEqual(single_label(scores)[0], "neutral")
        scores["happy"] = scores["sad"] = 1.0
        self.assertEqual(single_label(scores)[0:2], (None, "tied_max"))
        self.assertEqual(single_label(scores, tie_policy="exclude")[0:2], (None, "tied_max"))
        self.assertEqual(len(SINGLE_LABELS), 7)

    def test_baseline_verify_only_uses_saved_embeddings(self):
        with tempfile.TemporaryDirectory() as folder:
            source = Path(folder)
            (source / "indexes").mkdir()
            (source / "embeddings").mkdir()
            for split in ("train", "valid", "test"):
                (source / "indexes" / f"mosei_{split}_index.csv").touch()
                (source / "embeddings" / f"mosei_{split}.pt").touch()
            process = subprocess.run(
                [sys.executable, str(LAUNCHER), "--source-dir", str(source), "--verify-only"],
                cwd=ROOT, capture_output=True, text=True, check=True)
            config = json.loads(process.stdout)
            self.assertEqual(config["source_kind"], "baseline")
            self.assertEqual(config["splits"], ["train", "valid", "test"])
            self.assertEqual(config["tie_policy"], "exclude")

    def test_summary_aggregates_three_seeds(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            for seed, macro in ((42, 0.3), (43, 0.4), (44, 0.5)):
                path = root / "baseline" / f"seed_{seed}" / "metrics.json"
                path.parent.mkdir(parents=True)
                path.write_text(json.dumps({
                    "coverage": {"train": {"embedded": 9}, "validation": {"embedded": 3},
                                 "test": {"embedded": 3}},
                    "validation": {"accuracy": 0.5, "macro_f1": macro, "weighted_f1": 0.5},
                    "test": {"accuracy": 0.4, "macro_f1": macro - 0.1, "weighted_f1": 0.4},
                }))
            report = summarize_reports(root, ["baseline"], [42, 43, 44], "baseline", "first", False)
            self.assertAlmostEqual(
                report["conditions"]["baseline"]["metrics"]["validation"]["macro_f1"]["mean"], 0.4)
            self.assertEqual(report["conditions"]["baseline"]["coverage"]["test"]["embedded"], 3)


if __name__ == "__main__":
    unittest.main()
