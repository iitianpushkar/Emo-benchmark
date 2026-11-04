"""Check MOSEI predecode cache identity and resumable status files."""

import json
import tempfile
import unittest
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from cluster.mosei_decoded_cache import cache_available, cache_paths, identity, publish_cached
from cluster.predecode_mosei import chunk_state, finished_tasks, write_json, write_manifests
from cluster.mosei_video_decoder import TimedVideoDecoder


class MoseiPredecodeCacheTests(unittest.TestCase):
    def test_parallel_workers_finish_out_of_order(self):
        barrier = threading.Barrier(3, timeout=2)

        def decode(row):
            barrier.wait()
            return row["sample_id"]

        rows = [(i, {"sample_id": f"v[{i}]"}) for i in range(3)]
        with ThreadPoolExecutor(max_workers=3) as executor:
            completed = [(row["sample_id"], future.result()) for (_, row), future in
                         finished_tasks(rows, executor, decode, 3)]
        self.assertEqual(set(completed), {(f"v[{i}]", f"v[{i}]") for i in range(3)})

    def test_cache_ignores_prompt_but_detects_frame_change(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            video = root / "clip.mp4"
            video.write_bytes(b"video")
            protocol = {"fps": 6, "min_frames": 4, "max_frames": 64,
                        "frame_size": 224, "prompt_style": "utterance_only"}
            with patch("cluster.mosei_decoded_cache.qwen_utils_version", return_value="0.0.14"):
                expected = identity("v[0]", video, protocol)
                other_prompt = identity("v[0]", video,
                                        {**protocol, "prompt_style": "emotion_task"})
                other_size = identity("v[0]", video, {**protocol, "frame_size": 448})
            self.assertEqual(expected, other_prompt)
            self.assertNotEqual(expected, other_size)
            self.assertFalse(cache_available(root / "frames", expected))
            scratch = root / "scratch.pt"
            scratch.write_bytes(b"saved decoded tensor")
            publish_cached(root / "frames", expected, scratch)
            self.assertTrue(cache_available(root / "frames", expected))
            self.assertFalse(cache_available(root / "frames", other_size))
            self.assertTrue(scratch.exists())

    def test_chunk_state_and_partial_manifests(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            path = root / "chunks" / "train" / "train_00000_00002.json"
            state = chunk_state(path, ["v[0]", "v[1]"])
            state["success_ids"] = ["v[0]"]
            state["failed"] = {"v[1]": "TimeoutError: Decord exceeded 10s"}
            write_json(path, state)
            self.assertEqual(chunk_state(path, ["v[0]", "v[1]"]), state)
            with self.assertRaises(ValueError):
                chunk_state(path, ["v[1]", "v[0]"])
            summary = write_manifests(root, root / "frames",
                                      {"train": 2, "valid": 1, "test": 1})
            self.assertEqual(summary["successful"]["train"], 1)
            self.assertEqual(summary["failed"]["train"], 1)
            self.assertEqual(summary["pending"]["train"], 0)
            self.assertEqual(summary["pending"]["valid"], 1)
            self.assertEqual(json.loads((root / "summary.json").read_text())["expected"]["test"], 1)
            self.assertEqual(cache_paths(root / "frames", "v[0]")[0].suffix, ".pt")

    def test_cache_only_reader_never_starts_decord(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            video = root / "clip.mp4"
            video.write_bytes(b"video")
            protocol = {"fps": 6, "min_frames": 4, "max_frames": 64, "frame_size": 224}
            with patch("cluster.mosei_decoded_cache.qwen_utils_version", return_value="0.0.14"):
                expected = identity("v[0]", video, protocol)
                scratch = root / "scratch.pt"
                scratch.write_bytes(b"cached")
                publish_cached(root / "frames", expected, scratch)
                fake_frames = SimpleNamespace(shape=(4, 3, 224, 224))
                fake_torch = SimpleNamespace(load=lambda *_args, **_kwargs: (None, [(fake_frames, {})], {}))
                with patch.dict(sys.modules, {"torch": fake_torch}):
                    decoder = TimedVideoDecoder(10, cache_dir=root / "frames", cache_only=True)
                    try:
                        self.assertIs(decoder.decode([], "v[0]", video, protocol)[1][0][0], fake_frames)
                        with self.assertRaisesRegex(RuntimeError, "decoded cache entry is missing"):
                            decoder.decode([], "v[1]", video, protocol)
                        self.assertIsNone(decoder.process)
                    finally:
                        decoder.close()


if __name__ == "__main__":
    unittest.main()
