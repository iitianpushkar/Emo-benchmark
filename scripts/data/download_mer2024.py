#!/usr/bin/env python3
"""Download and extract the gated MER2024 labeled release from Hugging Face."""

from __future__ import annotations

import argparse
import os
import zipfile
from pathlib import Path


REPO_ID = "MERChallenge/MER2024"
FILES = ("video-labeled.zip", "label-transcription.csv", "label-disdim.csv")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("dataset/MER2024"))
    parser.add_argument("--keep-archive", action="store_true")
    return parser.parse_args()


def safe_extract(archive_path: Path, output_dir: Path) -> None:
    root = output_dir.resolve()
    with zipfile.ZipFile(archive_path) as archive:
        for member in archive.infolist():
            target = (output_dir / member.filename).resolve()
            if target != root and root not in target.parents:
                raise ValueError(f"Unsafe ZIP member: {member.filename}")
        archive.extractall(output_dir)


def main() -> None:
    try:
        from huggingface_hub import hf_hub_download
        from huggingface_hub.errors import GatedRepoError
    except ImportError as exc:
        raise SystemExit("Install the downloader with: python3 -m pip install huggingface_hub") from exc

    args = parse_args()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    downloaded: dict[str, Path] = {}
    try:
        for filename in FILES:
            print(f"Downloading {filename}", flush=True)
            downloaded[filename] = Path(
                hf_hub_download(
                    repo_id=REPO_ID,
                    repo_type="dataset",
                    filename=filename,
                    local_dir=output_dir,
                    token=os.environ.get("HF_TOKEN"),
                )
            )
    except GatedRepoError as exc:
        raise SystemExit(
            "MER2024 access is gated. Log in to Hugging Face, accept the dataset EULA, "
            "and run `hf auth login` or export an approved HF_TOKEN before retrying."
        ) from exc

    archive_path = downloaded["video-labeled.zip"]
    print(f"Extracting {archive_path}", flush=True)
    safe_extract(archive_path, output_dir)

    video_count = sum(1 for path in output_dir.rglob("*") if path.suffix.lower() == ".mp4")
    if video_count != 5030:
        raise RuntimeError(f"Expected 5,030 labeled MP4s after extraction, found {video_count}")

    if not args.keep_archive:
        archive_path.unlink()
    print(f"MER2024 labeled release ready at {output_dir} ({video_count:,} videos)")


if __name__ == "__main__":
    main()
