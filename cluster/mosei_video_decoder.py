"""Keep native video-decoder hangs outside the GPU model process."""

from __future__ import annotations

import multiprocessing as mp
import os
import tempfile
from pathlib import Path

from cluster.mosei_decoded_cache import identity, load_cached, publish_cached, validate_decoded


def _decode_loop(connection):
    os.environ["FORCE_QWENVL_VIDEO_READER"] = "decord"
    import torch
    from qwen_vl_utils import process_vision_info

    torch.set_num_threads(1)
    connection.send(("ready", None))
    try:
        while True:
            request = connection.recv()
            if request is None:
                break
            messages, output_path = request
            try:
                result = process_vision_info(
                    messages, image_patch_size=14, return_video_kwargs=True,
                    return_video_metadata=True,
                )
                torch.save(result, output_path)
                connection.send(("ok", None))
            except Exception as exc:
                connection.send(("error", f"{type(exc).__name__}: {exc}"))
    except EOFError:
        pass
    finally:
        connection.close()


class TimedVideoDecoder:
    """Reuse one CPU decoder process, replacing it after a per-clip timeout."""

    def __init__(self, timeout_seconds: float, cache_dir: Path | None = None,
                 cache_only: bool = False):
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        self.timeout_seconds = timeout_seconds
        self.cache_dir = Path(cache_dir).resolve() if cache_dir is not None else None
        self.cache_only = cache_only
        if cache_only and self.cache_dir is None:
            raise ValueError("cache_only requires cache_dir")
        self.context = mp.get_context("spawn")
        scratch = os.environ.get("SLURM_TMPDIR") or None
        self.tempdir = tempfile.TemporaryDirectory(prefix="mosei_decode_", dir=scratch)
        self.process = None
        self.connection = None
        self.sequence = 0

    def _start(self):
        parent, child = self.context.Pipe()
        process = self.context.Process(target=_decode_loop, args=(child,), daemon=True)
        process.start()
        child.close()
        self.process, self.connection = process, parent
        if not parent.poll(120):
            self._stop(force=True)
            raise RuntimeError("Video decoder worker did not start within 120 seconds")
        try:
            status, detail = parent.recv()
        except EOFError as exc:
            self._stop(force=True)
            raise RuntimeError("Video decoder worker exited during startup") from exc
        if status != "ready":
            self._stop(force=True)
            raise RuntimeError(f"Video decoder worker startup failed: {detail}")

    def _stop(self, force=False):
        process, connection = self.process, self.connection
        self.process, self.connection = None, None
        if process is not None:
            if process.is_alive():
                if force:
                    process.kill()
                else:
                    try:
                        connection.send(None)
                        process.join(timeout=2)
                    except (BrokenPipeError, OSError):
                        pass
                    if process.is_alive():
                        process.kill()
            process.join(timeout=5)
        if connection is not None:
            connection.close()

    def decode(self, messages, sample_id: str, video_path: Path | None = None,
               protocol: dict | None = None):
        import torch

        expected = None
        if self.cache_dir is not None:
            if video_path is None or protocol is None:
                raise ValueError("Cached decoding requires video_path and protocol")
            expected = identity(sample_id, video_path, protocol)
            try:
                cached = load_cached(self.cache_dir, expected)
            except Exception as exc:
                if self.cache_only:
                    raise RuntimeError(f"Video decoding failed for {sample_id}: "
                                       f"cached decode is invalid: {exc}") from exc
                cached = None
            if cached is not None:
                return cached
            if self.cache_only:
                raise RuntimeError(f"Video decoding failed for {sample_id}: "
                                   "decoded cache entry is missing")
        if self.process is None:
            self._start()
        self.sequence += 1
        output_path = Path(self.tempdir.name) / f"clip_{self.sequence:06d}.pt"
        try:
            self.connection.send((messages, str(output_path)))
            if not self.connection.poll(self.timeout_seconds):
                self._stop(force=True)
                raise TimeoutError(
                    f"Decord exceeded {self.timeout_seconds:g}s for {sample_id}; "
                    "decoder worker was killed and will restart"
                )
            try:
                status, detail = self.connection.recv()
            except EOFError as exc:
                self._stop(force=True)
                raise RuntimeError(f"Video decoder worker exited while reading {sample_id}") from exc
            if status != "ok":
                raise RuntimeError(f"Video decoding failed for {sample_id}: {detail}")
            result = torch.load(output_path, map_location="cpu", weights_only=False)
            if expected is not None:
                validate_decoded(result, protocol)
                publish_cached(self.cache_dir, expected, output_path)
            return result
        finally:
            output_path.unlink(missing_ok=True)

    def close(self):
        self._stop()
        self.tempdir.cleanup()
