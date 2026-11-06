"""Seven-class curve metrics and matched comparison inputs."""

import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import patch

from cluster.compare_mosei_token_curves import verify_match
from cluster import run_mosei_divprune_curve as curve
from cluster.run_mosei_divprune_curve import curve_ratios, write_curve


class CurveAlignmentTests(unittest.TestCase):
    def test_run_log_starts_before_model_and_mosei_verification(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            annotation = root / "datasets" / "mosei" / "csv" / "mosei_aligned.csv"
            annotation.parent.mkdir(parents=True)
            annotation.touch()
            events = []

            def log_started(_path):
                events.append("log")

            def verify_environment(**_kwargs):
                events.append("environment")
                return {}

            def verify_model(_path, _name):
                events.append("model")
                return {}

            def verify_mosei(_path, **_kwargs):
                events.append("mosei")
                raise RuntimeError("stop after verification begins")

            with (patch.object(sys, "argv", ["curve.py", "--root", str(root)]),
                  patch.object(curve, "activate_live_log", log_started),
                  patch.object(curve, "verify_qwen_environment", verify_environment),
                  patch.object(curve, "verify_model", verify_model),
                  patch.object(curve, "verify_mosei", verify_mosei)):
                with self.assertRaisesRegex(RuntimeError, "stop after verification begins"):
                    curve.main()
            self.assertEqual(events, ["log", "environment", "model", "mosei"])

    def test_seven_class_defaults_to_requested_ascending_grid_and_rejects_100_percent(self):
        ratios, names = curve_ratios(None, True)
        self.assertEqual(ratios, [.05, .1, .15, .2, .25, .3, .5, .7, .9])
        self.assertEqual(names, ["r005", "r010", "r015", "r020", "r025",
                                 "r030", "r050", "r070", "r090"])
        self.assertNotIn(1.0, ratios)
        with self.assertRaisesRegex(ValueError, "omit 1.0"):
            curve_ratios([1.0, 0.9], True)
        self.assertEqual(curve_ratios(None, False)[1][0], "r100")

    def test_seven_class_curve_uses_classification_macro_f1(self):
        class Axis:
            def errorbar(self, *_args, **_kwargs):
                pass

            def axhline(self, *_args, **_kwargs):
                pass

            def plot(self, *_args, **_kwargs):
                pass

            def set(self, **_kwargs):
                pass

            def grid(self, **_kwargs):
                pass

        class Axes:
            flat = [Axis() for _ in range(4)]

            def __getitem__(self, position):
                return self.flat[position[0] * 2 + position[1]]

        figure = SimpleNamespace(tight_layout=lambda: None, savefig=lambda *_args, **_kwargs: None)
        pyplot = SimpleNamespace(subplots=lambda *_args, **_kwargs: (figure, Axes()),
                                 close=lambda *_args: None)
        matplotlib = ModuleType("matplotlib")
        matplotlib.__path__ = []
        matplotlib.use = lambda *_args: None
        matplotlib.pyplot = pyplot
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / "run_config.json").write_text("{}")
            condition = root / "r090"
            mlp = condition / "mlp" / "seed_42"
            mlp.mkdir(parents=True)
            (condition / "coverage.json").write_text(json.dumps({
                "train": {"indexed": 3, "embedded": 3, "missing": 0},
                "valid": {"indexed": 2, "embedded": 2, "missing": 0},
                "test": {"indexed": 2, "embedded": 2, "missing": 0}}))
            valid_cost = {"weighted_retention": 1.0,
                          "mean_full_visual_tokens_per_clip": 100,
                          "mean_retained_visual_tokens_per_clip": 100,
                          "mean_retained_sequence_tokens_per_clip": 150,
                          "latency_seconds_per_clip": {
                              "end_to_end_seconds": {"mean": 1.0, "p95": 1.2},
                              "forward_seconds": {"mean": 0.8}},
                          "peak_total_allocated_gib": {"mean": 20},
                          "peak_total_reserved_gib": {"mean": 21},
                          "incremental_peak_allocated_gib": {"mean": 2},
                          "profiled_clips": 2,
                          "theoretical_forward_tflops_per_profiled_clip": {"mean": 1, "sd": 0}}
            (condition / "cost_summary.json").write_text(json.dumps({
                "train": {"weighted_retention": 1.0}, "valid": valid_cost,
                "test": {"weighted_retention": 1.0}}))
            (mlp / "metrics.json").write_text(json.dumps({
                "task": "single_max_seven_class",
                "validation": {"macro_f1": 0.4, "accuracy": 0.6,
                               "weighted_f1": 0.5}}))
            test_eval = condition / "test_eval" / "seed_42"
            test_eval.mkdir(parents=True)
            (test_eval / "metrics.json").write_text(json.dumps({
                "task": "single_max_seven_class",
                "metrics": {"macro_f1": 0.35, "accuracy": 0.55,
                            "weighted_f1": 0.45}}))
            with patch.dict(sys.modules, {"matplotlib": matplotlib, "matplotlib.pyplot": pyplot}):
                write_curve(root, ["r090"], (42,), single_label_mode=True)
            row = json.loads((root / "validation_curve.json").read_text())[0]
            self.assertEqual(row["valid_macro_f1_mean"], 0.4)
            self.assertEqual(row["valid_accuracy_mean"], 0.6)
            self.assertEqual(row["test_macro_f1_mean"], 0.35)
            self.assertEqual(row["test_accuracy_mean"], 0.55)
            self.assertNotIn("valid_macro_f1_at_0.5_mean", row)
            self.assertNotIn("delta_valid_macro_f1", row)

    def test_rejects_different_cached_subset_or_prompt(self):
        config = {"model_config_sha256": "model", "model_index_sha256": "weights",
                  "aligned_csv_sha256": "selected-clips", "ground_truth_csv_sha256": "labels",
                  "target": "single highest emotion", "prompt_style": "emotion_task",
                  "cache_selection": {"selected": {"train": 10, "valid": 3, "test": 4}},
                  "protocol": {"fps": 6, "prompt_style": "emotion_task"},
                  "seeds": [42, 43, 44], "measurement_environment": {"gpu": "H100"}}
        rows = {1.0: {"valid_macro_f1_mean": 0.5}}
        costs = {1.0: {"profiled_sample_ids": ["clip-a"],
                       "full_visual_tokens": 100, "retained_visual_tokens": 100}}
        run = (dict(config), rows, costs)
        verify_match(Path("random"), run, config, rows, costs)
        run[0]["aligned_csv_sha256"] = "different-clips"
        with self.assertRaisesRegex(ValueError, "aligned_csv_sha256"):
            verify_match(Path("random"), run, config, rows, costs)
        run[0]["aligned_csv_sha256"] = "selected-clips"
        run[0]["prompt_style"] = "utterance_only"
        with self.assertRaisesRegex(ValueError, "prompt_style"):
            verify_match(Path("random"), run, config, rows, costs)


if __name__ == "__main__":
    unittest.main()
