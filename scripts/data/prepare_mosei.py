#!/usr/bin/env python3
"""Build verified CMU-MOSEI indexes for six binary emotion targets."""

from __future__ import annotations

import argparse
import csv
import math
from collections import Counter
from pathlib import Path

EMOTIONS = ("happy", "sad", "anger", "surprise", "disgust", "fear")
SPLITS = ("train", "valid", "test")
REQUIRED = {"sample_id", "video_id", "clip_id", "split", "text", "start_seconds", "end_seconds", *EMOTIONS}
FIELDS = (
    "sample_id", "split", "video_id", "clip_id", "video_path", "video_exists",
    "utterance", "emotion", "emotion_id", "start_seconds", "end_seconds", "sentiment", *EMOTIONS,
    *(f"{name}_label" for name in EMOTIONS),
)


def prepare_rows(aligned_csv: Path, raw_dir: Path, include_missing: bool = False) -> tuple[dict[str, list[dict[str, str]]], Counter]:
    rows = {split: [] for split in SPLITS}
    missing = Counter()
    seen_samples: set[str] = set()
    video_splits: dict[str, str] = {}
    with aligned_csv.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        absent = REQUIRED - set(reader.fieldnames or ())
        if absent:
            raise ValueError(f"Missing columns: {sorted(absent)}")
        for line, source in enumerate(reader, start=2):
            sample_id = source["sample_id"]
            split = source["split"]
            video_id = source["video_id"]
            clip_id = source["clip_id"]
            if split not in rows:
                raise ValueError(f"Line {line}: unknown split {split!r}")
            if sample_id in seen_samples:
                raise ValueError(f"Line {line}: duplicate sample_id {sample_id!r}")
            seen_samples.add(sample_id)
            if sample_id != f"{video_id}[{int(clip_id)}]":
                raise ValueError(f"Line {line}: sample_id does not match video/clip")
            if video_id in video_splits and video_splits[video_id] != split:
                raise ValueError(f"Line {line}: video {video_id!r} crosses splits")
            video_splits[video_id] = split
            start, end = float(source["start_seconds"]), float(source["end_seconds"])
            # CSD intervals can start slightly below zero; the distributed MP4
            # is already cut to the segment, so retain the original timestamp.
            if not (math.isfinite(start) and math.isfinite(end) and start < end):
                raise ValueError(f"Line {line}: invalid segment interval")
            video_path = raw_dir / video_id / f"{clip_id}.mp4"
            exists = video_path.is_file()
            if not exists:
                missing[split] += 1
                if not include_missing:
                    continue
            row = {
                "sample_id": sample_id,
                "split": split,
                "video_id": video_id,
                "clip_id": clip_id,
                "video_path": str(video_path.absolute()),
                "video_exists": str(exists),
                "utterance": " ".join((source["text"] or "").split()),
                # Legacy extractors require a scalar label. MOSEI training must
                # join the six *_label columns by sample_id; this value is unused.
                "emotion": "multilabel_placeholder",
                "emotion_id": "0",
                "start_seconds": source["start_seconds"],
                "end_seconds": source["end_seconds"],
                "sentiment": source.get("sentiment", ""),
            }
            for emotion in EMOTIONS:
                score = float(source[emotion])
                if not math.isfinite(score):
                    raise ValueError(f"Line {line}: invalid {emotion} score")
                row[emotion] = source[emotion]
                row[f"{emotion}_label"] = str(int(score > 0))
            rows[split].append(row)
    return rows, missing


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mosei-root", type=Path, default=Path("dataset/CMU-MOSEI"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/indexes"))
    parser.add_argument("--include-missing", action="store_true")
    args = parser.parse_args()
    root = args.mosei_root.expanduser().resolve()
    rows, missing = prepare_rows(root / "csv" / "mosei_aligned.csv", root / "Raw", args.include_missing)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for split in SPLITS:
        path = args.output_dir / f"mosei_{split}_index.csv"
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=FIELDS)
            writer.writeheader()
            writer.writerows(rows[split])
        counts = {name: sum(int(row[f"{name}_label"]) for row in rows[split]) for name in EMOTIONS}
        print(f"{split}: {len(rows[split])} rows, {missing[split]} missing clips, positives={counts} -> {path}")


if __name__ == "__main__":
    main()
