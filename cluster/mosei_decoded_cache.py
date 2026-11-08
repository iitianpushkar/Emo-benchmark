"""Portable lookup and integrity metadata for sampled Qwen video frames."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import shutil
from functools import lru_cache
from pathlib import Path

SCHEMA = 1
FRAME_KEYS = ("fps", "min_frames", "max_frames", "frame_size")


@lru_cache(maxsize=1)
def qwen_utils_version() -> str:
    return importlib.metadata.version("qwen-vl-utils")


def cache_paths(root: Path, sample_id: str) -> tuple[Path, Path]:
    if not sample_id:
        raise ValueError("sample_id must be nonempty")
    digest = hashlib.sha256(sample_id.encode("utf-8")).hexdigest()
    folder = root / digest[:2]
    return folder / f"{digest}.pt", folder / f"{digest}.json"


def identity(sample_id: str, video_path: Path, protocol: dict) -> dict:
    video_path = video_path.resolve(strict=True)
    stat = video_path.stat()
    return {
        "schema": SCHEMA,
        "sample_id": sample_id,
        "video_path": str(video_path),
        "video_bytes": stat.st_size,
        "video_mtime_ns": stat.st_mtime_ns,
        "decoder": "qwen_vl_utils.process_vision_info/decord",
        "qwen_vl_utils": qwen_utils_version(),
        "image_patch_size": 14,
        "frame_protocol": {key: protocol[key] for key in FRAME_KEYS},
    }


def cache_available(root: Path, expected: dict) -> bool:
    data_path, metadata_path = cache_paths(root, expected["sample_id"])
    if not data_path.is_file() or data_path.stat().st_size == 0 or not metadata_path.is_file():
        return False
    try:
        return json.loads(metadata_path.read_text(encoding="utf-8")) == expected
    except (OSError, ValueError):
        return False


def validate_decoded(result, protocol: dict) -> None:
    if not isinstance(result, tuple) or len(result) != 3 or not result[1]:
        raise ValueError("Decoder did not return a video result")
    if not isinstance(result[2], dict):
        raise ValueError("Decoder video kwargs are invalid")
    frames = result[1][0][0]
    shape = tuple(frames.shape)
    if (len(shape) != 4 or not 1 <= shape[0] <= protocol["max_frames"]
            or shape[-2:] != (protocol["frame_size"], protocol["frame_size"])):
        raise ValueError(f"Unexpected decoded video shape: {shape}")


def load_cached(root: Path, expected: dict):
    if not cache_available(root, expected):
        return None
    import torch

    data_path, _ = cache_paths(root, expected["sample_id"])
    result = torch.load(data_path, map_location="cpu", weights_only=False)
    validate_decoded(result, expected["frame_protocol"])
    return result


def publish_cached(root: Path, expected: dict, decoded_file: Path) -> None:
    data_path, metadata_path = cache_paths(root, expected["sample_id"])
    data_path.parent.mkdir(parents=True, exist_ok=True)
    data_tmp = data_path.with_suffix(".pt.tmp")
    metadata_tmp = metadata_path.with_suffix(".json.tmp")
    try:
        # SLURM_TMPDIR may be a different filesystem from the shared cache.
        shutil.copyfile(decoded_file, data_tmp)
        data_tmp.replace(data_path)
        metadata_tmp.write_text(json.dumps(expected, sort_keys=True) + "\n", encoding="utf-8")
        metadata_tmp.replace(metadata_path)
    finally:
        data_tmp.unlink(missing_ok=True)
        metadata_tmp.unlink(missing_ok=True)
