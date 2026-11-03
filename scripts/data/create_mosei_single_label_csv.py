#!/usr/bin/env python3
"""Create a seven-class CMU-MOSEI CSV directly from aligned annotations."""

from __future__ import annotations

import argparse
import csv
import math
import sys
from collections import Counter
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.data.prepare_mosei import EMOTIONS, SINGLE_LABELS, SPLITS, single_label

REQUIRED = {"sample_id", "video_id", "clip_id", "split", "text", *EMOTIONS}
ADDED = ("utterance", "emotion", "emotion_id", "label_status", "max_emotion_score")


def create_csv(source: Path, output: Path) -> dict[str, dict[str, int]]:
    if source.resolve() == output.resolve():
        raise ValueError("Output must differ from the source annotation CSV")
    counts = {split: Counter() for split in SPLITS}
    seen = set()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name + ".tmp")
    try:
        with source.open(newline="", encoding="utf-8") as source_file, temporary.open(
            "w", newline="", encoding="utf-8"
        ) as output_file:
            reader = csv.DictReader(source_file)
            missing = REQUIRED - set(reader.fieldnames or ())
            if missing:
                raise ValueError(f"{source}: missing columns {sorted(missing)}")
            if set(ADDED) & set(reader.fieldnames or ()):
                raise ValueError("Source already contains a derived single-label column")
            writer = csv.DictWriter(output_file, fieldnames=[*(reader.fieldnames or ()), *ADDED])
            writer.writeheader()
            for line, row in enumerate(reader, start=2):
                split, sample_id = row["split"], row["sample_id"]
                if split not in counts:
                    raise ValueError(f"Line {line}: unknown split {split!r}")
                if sample_id in seen:
                    raise ValueError(f"Line {line}: duplicate sample ID {sample_id!r}")
                if sample_id != f"{row['video_id']}[{int(row['clip_id'])}]":
                    raise ValueError(f"Line {line}: sample ID does not match video and clip IDs")
                seen.add(sample_id)
                scores = {name: float(row[name]) for name in EMOTIONS}
                if not all(math.isfinite(value) and value >= 0 for value in scores.values()):
                    raise ValueError(f"Line {line}: emotion scores must be finite and nonnegative")
                label, status, maximum, _ = single_label(scores, "exclude", "neutral")
                counts[split]["source"] += 1
                counts[split][status] += 1
                if label is None:
                    continue
                writer.writerow({**row, "utterance": " ".join(row["text"].split()),
                                 "emotion": label, "emotion_id": SINGLE_LABELS.index(label),
                                 "label_status": status, "max_emotion_score": maximum})
                counts[split]["written"] += 1
                counts[split][label] += 1
        temporary.replace(output)
    finally:
        temporary.unlink(missing_ok=True)
    return {split: dict(counts[split]) for split in SPLITS}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path,
                        default=Path("dataset/CMU-MOSEI/csv/mosei_aligned.csv"))
    parser.add_argument("--output", type=Path,
                        default=Path("dataset/CMU-MOSEI/csv/mosei_single_label.csv"))
    args = parser.parse_args()
    summary = create_csv(args.source, args.output)
    for split, counts in summary.items():
        print(f"{split}: {counts}")
    print(f"Seven-class CSV: {args.output}")


if __name__ == "__main__":
    main()
