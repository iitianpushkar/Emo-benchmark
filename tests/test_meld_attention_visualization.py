"""Single-clip full-baseline attention visualization inputs and frame mapping."""

import csv
import contextlib
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import patch

import numpy as np

from cluster import visualize_meld_full_attention as visualize


class MeldAttentionVisualizationTests(unittest.TestCase):
    def test_temporal_groups_cover_each_decoded_frame(self):
        self.assertEqual(visualize.frame_group_indices(4, 2, 2), [0, 0, 1, 1])
        with self.assertRaisesRegex(ValueError, "do not match"):
            visualize.frame_group_indices(5, 2, 2)

    def test_one_overlay_per_layer_contains_every_frame(self):
        shown = []
        saved = []

        class Axis:
            def imshow(self, image, **_kwargs):
                shown.append(tuple(image.shape))

            def set_title(self, *_args, **_kwargs):
                pass

            def set_xticks(self, *_args):
                pass

            def set_yticks(self, *_args):
                pass

            def axis(self, *_args):
                pass

        axes = SimpleNamespace(flat=[Axis() for _ in range(4)])
        figure = SimpleNamespace(suptitle=lambda *_args, **_kwargs: None,
                                 tight_layout=lambda **_kwargs: None,
                                 savefig=lambda path, **_kwargs: saved.append(path.name))
        pyplot = SimpleNamespace(subplots=lambda *_args, **_kwargs: (figure, axes),
                                 close=lambda *_args: None)
        matplotlib = ModuleType("matplotlib")
        matplotlib.__path__ = []
        matplotlib.use = lambda *_args: None
        matplotlib.pyplot = pyplot
        with tempfile.TemporaryDirectory() as folder, patch.dict(
            sys.modules, {"matplotlib": matplotlib, "matplotlib.pyplot": pyplot}
        ):
            visualize.save_every_frame_overlays(
                Path(folder), np.zeros((4, 8, 8, 3), dtype=np.uint8),
                np.ones((2, 2, 2, 2), dtype=np.float32), 2)
        self.assertEqual(saved, ["layer_00.png", "layer_01.png"])
        self.assertEqual(shown.count((8, 8, 3)), 8)
        self.assertEqual(shown.count((2, 2)), 8)

    def test_verify_only_selects_one_cached_clip_without_loading_gpu(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            baseline = root / "results" / "qwen2_5_vl_7b_meld_emotion_prompt_cache_success"
            index = baseline / "indexes" / "meld_test_index.csv"
            index.parent.mkdir(parents=True)
            video = root / "clip.mp4"
            video.touch()
            with index.open("w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=("sample_id", "video_path", "utterance"))
                writer.writeheader()
                writer.writerow({"sample_id": "test_dia1_utt0", "video_path": str(video),
                                 "utterance": "hello"})
            cache = root / "datasets" / "meld" / "decoded_qwen_6fps_224_v1"
            (cache / "frames").mkdir(parents=True)
            (cache / "successful_decodes.csv").write_text("split,sample_id,cache_path\n")
            argv = ["visualize.py", "--root", str(root), "--sample-id",
                    "test_dia1_utt0", "--verify-only"]
            with (patch.object(sys, "argv", argv),
                  patch.object(visualize, "identity", return_value={}),
                  patch.object(visualize, "cache_available", return_value=True),
                  patch.object(visualize, "verify_model", return_value={"model": "7b"})):
                self.assertEqual(visualize.main(), 0)

    def test_mosei_verify_only_selects_matching_baseline_and_prompt(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            cache = root / "datasets" / "mosei" / "decoded_qwen_6fps_224_v1"
            (cache / "frames").mkdir(parents=True)
            (cache / "successful_decodes.csv").write_text("split,sample_id,cache_path\n")
            video = root / "clip.mp4"
            video.touch()
            for mode, suffix in (("emotion_multilabel_task", "multilabel"),
                                 ("emotion_task", "single_label")):
                baseline = (root / "results" /
                            f"qwen2_5_vl_7b_mosei_{suffix}_emotion_prompt_cache_success")
                index = baseline / "indexes" / "mosei_test_index.csv"
                index.parent.mkdir(parents=True)
                with index.open("w", newline="") as handle:
                    writer = csv.DictWriter(handle, fieldnames=("sample_id", "video_path", "utterance"))
                    writer.writeheader()
                    writer.writerow({"sample_id": "clip[0]", "video_path": str(video),
                                     "utterance": "hello"})
                argv = ["visualize_mosei.py", "--root", str(root), "--sample-id",
                        "clip[0]", "--prompt-style", mode, "--verify-only"]
                with (patch.object(sys, "argv", argv),
                      patch.object(visualize, "identity", return_value={}),
                      patch.object(visualize, "cache_available", return_value=True),
                      patch.object(visualize, "verify_model", return_value={"model": "7b"}),
                      contextlib.redirect_stdout(io.StringIO()) as output):
                    self.assertEqual(visualize.main("mosei"), 0)
                report = json.loads(output.getvalue())
                self.assertEqual(report["prompt_style"], mode)
                self.assertEqual(report["index_csv"], str(index.resolve()))


if __name__ == "__main__":
    unittest.main()
