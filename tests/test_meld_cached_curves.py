"""MELD cached-frame selection, prompt, labels, and curve defaults."""

import csv
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import patch

import torch

from cluster import meld_cache_subset, run_meld_qwen2_5vl7b as meld_run
from cluster.extract_mosei_qwen2_5vl_multi import chunk_config
from cluster.meld_prompt import make_meld_prompt


class MeldCachedCurveTests(unittest.TestCase):
    def test_cache_selection_and_label_attachment(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            indexes = root / "indexes_all"
            indexes.mkdir()
            cache = root / "cache" / "frames"
            cache.mkdir(parents=True)
            manifest_rows = []
            for split, label in (("train", 3), ("dev", 5), ("test", 0)):
                sample_id = f"{split}_dia1_utt0"
                video = root / f"{split}.mp4"
                video.touch()
                with (indexes / f"meld_{split}_index.csv").open("w", newline="") as handle:
                    writer = csv.DictWriter(handle, fieldnames=("sample_id", "split", "video_path",
                                                                "video_exists", "emotion_id", "utterance"))
                    writer.writeheader()
                    writer.writerow({"sample_id": sample_id, "split": split,
                                     "video_path": str(video), "video_exists": "True",
                                     "emotion_id": label, "utterance": "hello"})
                manifest_rows.append({"split": split, "sample_id": sample_id,
                                      "cache_path": str(meld_cache_subset.cache_paths(cache, sample_id)[0])})
                data_path, metadata_path = meld_cache_subset.cache_paths(cache, sample_id)
                data_path.parent.mkdir(parents=True, exist_ok=True)
                data_path.write_bytes(b"cached frames")
                with patch("cluster.mosei_decoded_cache.qwen_utils_version", return_value="test"):
                    expected = meld_cache_subset.identity(sample_id, video, {
                        "fps": 6, "min_frames": 4, "max_frames": 64, "frame_size": 224})
                metadata_path.write_text(json.dumps(expected))
            with (cache.parent / "successful_decodes.csv").open("w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=("split", "sample_id", "cache_path"))
                writer.writeheader()
                writer.writerows(manifest_rows)
            with patch("cluster.mosei_decoded_cache.qwen_utils_version", return_value="test"):
                rows, summary = meld_cache_subset.selected_cached_rows(indexes, cache, {
                    "fps": 6, "min_frames": 4, "max_frames": 64, "frame_size": 224})
            self.assertEqual(summary["selected"], {"train": 1, "dev": 1, "test": 1})
            selected_indexes = root / "indexes"
            meld_cache_subset.write_selected_indexes(selected_indexes, rows)
            embedding = root / "train.pt"
            torch.save({"sample_ids": ["train_dia1_utt0"],
                        "embeddings": torch.ones(1, 3584), "labels": torch.zeros(1)}, embedding)
            meld_cache_subset.attach_labels(embedding, selected_indexes / "meld_train_index.csv")
            payload = torch.load(embedding, weights_only=False)
            self.assertEqual(payload["labels"].tolist(), [3])
            self.assertEqual(payload["label_names"][3], "joy")

    def test_meld_prompt_and_chunk_marker(self):
        prompt = make_meld_prompt("hello")
        self.assertIn("joy, neutral, sadness", prompt)
        self.assertNotIn("happy", prompt)
        args = SimpleNamespace(model_dir=Path("/models/qwen"), start=0, limit=2,
                               pruning="divprune", profile_stride=1,
                               prompt_style="meld_emotion_task")
        config = chunk_config(args, .5, {"fps": 6})
        self.assertEqual(config["prompt_style"], "meld_seven_class_emotion_task_v1")

    def test_verify_only_uses_seven_ratios_and_cache_counts(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            meld = root / "datasets" / "meld"
            meld.mkdir(parents=True)
            cache = meld / "decoded_qwen_6fps_224_v1" / "frames"
            cache.mkdir(parents=True)
            (cache.parent / "successful_decodes.csv").write_text("split,sample_id,cache_path\n")
            rows = {split: [{"sample_id": f"{split}_0"}] for split in ("train", "dev", "test")}
            summary = {"selected": dict.fromkeys(rows, 1), "stale_success_entries": 0}
            with (patch.object(sys, "argv", ["curve.py", "--root", str(root), "--verify-only"]),
                  patch.object(meld_run, "verify_qwen_environment", return_value={}),
                  patch.object(meld_run, "verify_model", return_value={}),
                  patch.object(meld_run, "verify_meld", return_value={"root": str(meld)}),
                  patch.object(meld_run, "run"),
                  patch.object(meld_run, "selected_cached_rows", return_value=(rows, summary))):
                self.assertEqual(meld_run.main("divprune"), 0)
                self.assertEqual(meld_run.main(), 0)

    def test_test_metric_curve_summarizes_seeds_and_writes_png(self):
        class Axis:
            def errorbar(self, *_args, **_kwargs):
                pass

            def set(self, **_kwargs):
                pass

            def grid(self, **_kwargs):
                pass

        class Axes:
            flat = [Axis() for _ in range(4)]

            def __getitem__(self, position):
                return self.flat[position[0] * 2 + position[1]]

        saved = []
        figure = SimpleNamespace(tight_layout=lambda: None,
                                 savefig=lambda path, **_kwargs: saved.append(path.name))
        pyplot = SimpleNamespace(subplots=lambda *_args, **_kwargs: (figure, Axes()),
                                 close=lambda *_args: None)
        matplotlib = ModuleType("matplotlib")
        matplotlib.__path__ = []
        matplotlib.use = lambda *_args: None
        matplotlib.pyplot = pyplot
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            ratio = root / "r070"
            ratio.mkdir()
            (ratio / "coverage.json").write_text(json.dumps({
                split: {"indexed": 2, "embedded": 2, "missing": 0}
                for split in ("train", "dev", "test")}))
            cost = {"weighted_retention": 0.7, "mean_retained_visual_tokens_per_clip": 70,
                    "latency_seconds_per_clip": {"end_to_end_seconds": {"mean": 1.0}},
                    "peak_total_allocated_gib": {"mean": 12},
                    "theoretical_forward_tflops_per_profiled_clip": {"mean": 1.5}}
            (ratio / "cost_summary.json").write_text(json.dumps({"dev": cost, "test": cost}))
            for seed, f1 in ((42, 0.4), (43, 0.6)):
                mlp = ratio / "mlp" / f"seed_{seed}"
                mlp.mkdir(parents=True)
                (mlp / "metrics.json").write_text(json.dumps({
                    "best_dev": {"macro_f1": f1, "accuracy": 0.7, "weighted_f1": 0.6},
                    "test": {"macro_f1": f1, "accuracy": 0.65, "weighted_f1": 0.55}}))
            with patch.dict(sys.modules, {"matplotlib": matplotlib,
                                          "matplotlib.pyplot": pyplot}):
                rows = meld_run.write_curve(root, ["r070"], [42, 43], "divprune")
            self.assertEqual(rows[0]["test_macro_f1_mean"], 0.5)
            self.assertEqual(rows[0]["test_accuracy_mean"], 0.65)
            self.assertEqual(saved, ["test_metrics_curve.png"])
            self.assertTrue((root / "validation_curve.csv").is_file())


if __name__ == "__main__":
    unittest.main()
