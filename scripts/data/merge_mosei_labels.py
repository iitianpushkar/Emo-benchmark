#!/usr/bin/env python3
"""Merge processed CMU-MOSEI labels with CSD timestamps and emotion labels."""

from __future__ import annotations

import argparse
import re
from difflib import SequenceMatcher
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment


EMOTION_COLUMNS = ["happy", "sad", "anger", "surprise", "disgust", "fear"]
SENTIMENT_TOLERANCE = 1e-6


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels-csv", type=Path, default=Path("dataset/CMU-MOSEI/label.csv"))
    parser.add_argument(
        "--segments-csv",
        type=Path,
        default=Path("dataset/CMU-MOSEI/csv/mosei_segments.csv"),
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=Path("dataset/CMU-MOSEI/csv/mosei_aligned.csv"),
    )
    return parser.parse_args()


def normalize_text(value: object) -> str:
    return " ".join(re.findall(r"[a-z0-9]+", str(value).lower()))


def text_similarity(left: object, right: object) -> float:
    return SequenceMatcher(None, normalize_text(left), normalize_text(right), autojunk=False).ratio()


def align_video(labels: pd.DataFrame, segments: pd.DataFrame) -> list[dict[str, object]]:
    labels = labels.copy()
    segments = segments.sort_values("segment_index").reset_index(drop=True)
    labels["clip_sort"] = labels["clip_id"].astype(str)
    labels = labels.sort_values("clip_sort").reset_index(drop=True)

    if len(labels) == len(segments):
        pairs = list(zip(range(len(labels)), range(len(segments)), strict=True))
        method = "clip_id_lexicographic_order"
    else:
        score = np.full((len(labels), len(segments)), -1_000.0)
        for label_index, label_row in labels.iterrows():
            for segment_index, segment_row in segments.iterrows():
                if abs(float(label_row["label"]) - float(segment_row["sentiment"])) <= SENTIMENT_TOLERANCE:
                    score[label_index, segment_index] = text_similarity(
                        label_row["text"], segment_row["transcript"]
                    )
        label_indices, segment_indices = linear_sum_assignment(-score)
        pairs = list(zip(label_indices.tolist(), segment_indices.tolist(), strict=True))
        method = "transcript_similarity"

    aligned: list[dict[str, object]] = []
    for label_index, segment_index in pairs:
        label_row = labels.iloc[label_index]
        segment_row = segments.iloc[segment_index]
        difference = abs(float(label_row["label"]) - float(segment_row["sentiment"]))
        if difference > SENTIMENT_TOLERANCE:
            raise ValueError(
                f"Sentiment mismatch for {label_row['video_id']} clip {label_row['clip_id']}: "
                f"{label_row['label']} versus {segment_row['sentiment']}"
            )
        aligned.append(
            {
                "sample_id": f"{label_row['video_id']}[{int(label_row['clip_id'])}]",
                "video_id": label_row["video_id"],
                "clip_id": int(label_row["clip_id"]),
                "split": label_row["mode"],
                "text": label_row["text"],
                "sentiment": float(label_row["label"]),
                "sentiment_annotation": label_row["annotation"],
                "start_seconds": float(segment_row["start_seconds"]),
                "end_seconds": float(segment_row["end_seconds"]),
                **{column: float(segment_row[column]) for column in EMOTION_COLUMNS},
                "word_count": int(segment_row["word_count"]),
                "reconstructed_transcript": segment_row["transcript"],
                "transcript_similarity": text_similarity(label_row["text"], segment_row["transcript"]),
                "csd_segment_index": int(segment_row["segment_index"]),
                "alignment_method": method,
            }
        )
    return aligned


def merge_labels(labels_csv: Path, segments_csv: Path) -> pd.DataFrame:
    labels = pd.read_csv(labels_csv)
    segments = pd.read_csv(segments_csv)
    required_labels = {"video_id", "clip_id", "text", "label", "annotation", "mode"}
    required_segments = {
        "video_id",
        "segment_index",
        "start_seconds",
        "end_seconds",
        "sentiment",
        *EMOTION_COLUMNS,
        "word_count",
        "transcript",
    }
    if not required_labels.issubset(labels.columns):
        raise ValueError(f"Missing label columns: {sorted(required_labels - set(labels.columns))}")
    if not required_segments.issubset(segments.columns):
        raise ValueError(f"Missing segment columns: {sorted(required_segments - set(segments.columns))}")
    if labels.duplicated(["video_id", "clip_id"]).any():
        raise ValueError("Processed labels contain duplicate video_id/clip_id pairs")

    segment_groups = {video_id: group for video_id, group in segments.groupby("video_id", sort=False)}
    aligned: list[dict[str, object]] = []
    for video_id, label_group in labels.groupby("video_id", sort=True):
        if video_id not in segment_groups:
            raise ValueError(f"No CSD segments found for video {video_id}")
        aligned.extend(align_video(label_group, segment_groups[video_id]))

    output = pd.DataFrame(aligned).sort_values(["split", "video_id", "clip_id"]).reset_index(drop=True)
    if len(output) != len(labels):
        raise RuntimeError(f"Expected {len(labels):,} aligned rows, produced {len(output):,}")
    if output["sample_id"].duplicated().any():
        raise RuntimeError("Aligned output contains duplicate sample IDs")
    if output["split"].isna().any():
        raise RuntimeError("Aligned output contains missing split assignments")
    return output


def main() -> None:
    args = parse_args()
    labels_csv = args.labels_csv.expanduser().resolve()
    segments_csv = args.segments_csv.expanduser().resolve()
    output_csv = args.output_csv.expanduser().resolve()
    output = merge_labels(labels_csv, segments_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(output_csv, index=False)
    print(f"Aligned {len(output):,} CMU-MOSEI segments -> {output_csv}")
    print(output["split"].value_counts().reindex(["train", "valid", "test"]).to_string())
    print("Alignment methods:")
    print(output["alignment_method"].value_counts().to_string())


if __name__ == "__main__":
    main()
