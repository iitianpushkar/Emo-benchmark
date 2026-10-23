#!/usr/bin/env python3
"""Export CMU-MOSEI label and timestamped-word CSD files to CSV."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import h5py
import numpy as np


LABEL_NAMES = ["sentiment", "happy", "sad", "anger", "surprise", "disgust", "fear"]
SILENCE_TOKENS = {"sp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--metadata-dir",
        type=Path,
        default=Path("dataset/CMU-MOSEI/metadata"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("dataset/CMU-MOSEI/csv"),
    )
    return parser.parse_args()


def decode_word(value: object) -> str:
    if isinstance(value, np.ndarray):
        value = value.reshape(-1)[0]
    if isinstance(value, (bytes, np.bytes_)):
        return bytes(value).decode("utf-8", errors="replace")
    return str(value)


def format_number(value: float) -> str:
    return format(float(value), ".10g")


def export_csd(metadata_dir: Path, output_dir: Path) -> dict[str, int]:
    labels_path = metadata_dir / "CMU_MOSEI_Labels.csd"
    words_path = metadata_dir / "CMU_MOSEI_TimestampedWords.csd"
    if not labels_path.is_file() or not words_path.is_file():
        raise FileNotFoundError("CMU-MOSEI label and timestamped-word CSD files are required")

    output_dir.mkdir(parents=True, exist_ok=True)
    labels_csv = output_dir / "mosei_labels.csv"
    words_csv = output_dir / "mosei_timestamped_words.csv"
    segments_csv = output_dir / "mosei_segments.csv"

    label_columns = [
        "video_id",
        "segment_index",
        "segment_id",
        "start_seconds",
        "end_seconds",
        *LABEL_NAMES,
    ]
    word_columns = ["video_id", "word_index", "word", "start_seconds", "end_seconds", "is_silence"]
    segment_columns = [*label_columns, "word_count", "transcript"]

    label_count = 0
    word_count = 0
    transcript_word_count = 0
    segments_without_words = 0

    with (
        h5py.File(labels_path, "r") as labels_file,
        h5py.File(words_path, "r") as words_file,
        labels_csv.open("w", newline="", encoding="utf-8") as labels_handle,
        words_csv.open("w", newline="", encoding="utf-8") as words_handle,
        segments_csv.open("w", newline="", encoding="utf-8") as segments_handle,
    ):
        labels_data = labels_file["All Labels"]["data"]
        words_data = words_file["words"]["data"]
        labels_writer = csv.writer(labels_handle)
        words_writer = csv.writer(words_handle)
        segments_writer = csv.writer(segments_handle)
        labels_writer.writerow(label_columns)
        words_writer.writerow(word_columns)
        segments_writer.writerow(segment_columns)

        decoded_words: dict[str, tuple[list[str], np.ndarray]] = {}
        for video_id in sorted(words_data.keys()):
            group = words_data[video_id]
            features = group["features"][:]
            intervals = group["intervals"][:]
            tokens = [decode_word(value).strip() for value in features]
            decoded_words[video_id] = (tokens, intervals)
            for index, (token, interval) in enumerate(zip(tokens, intervals, strict=True)):
                words_writer.writerow(
                    [
                        video_id,
                        index,
                        token,
                        format_number(interval[0]),
                        format_number(interval[1]),
                        token.lower() in SILENCE_TOKENS,
                    ]
                )
                word_count += 1

        for video_id in sorted(labels_data.keys()):
            group = labels_data[video_id]
            features = group["features"][:]
            intervals = group["intervals"][:]
            tokens, word_intervals = decoded_words.get(video_id, ([], np.empty((0, 2))))
            for index, (label_values, interval) in enumerate(zip(features, intervals, strict=True)):
                segment_id = f"{video_id}[{index}]"
                label_row = [
                    video_id,
                    index,
                    segment_id,
                    format_number(interval[0]),
                    format_number(interval[1]),
                    *[format_number(value) for value in label_values],
                ]
                labels_writer.writerow(label_row)
                label_count += 1

                selected_tokens = [
                    token
                    for token, word_interval in zip(tokens, word_intervals, strict=True)
                    if word_interval[1] > interval[0]
                    and word_interval[0] < interval[1]
                    and token.lower() not in SILENCE_TOKENS
                ]
                if not selected_tokens:
                    segments_without_words += 1
                transcript_word_count += len(selected_tokens)
                segments_writer.writerow([*label_row, len(selected_tokens), " ".join(selected_tokens)])

    return {
        "labels": label_count,
        "words": word_count,
        "transcript_words": transcript_word_count,
        "segments_without_words": segments_without_words,
    }


def main() -> None:
    args = parse_args()
    counts = export_csd(args.metadata_dir.expanduser().resolve(), args.output_dir.expanduser().resolve())
    print(f"Exported {counts['labels']:,} labels")
    print(f"Exported {counts['words']:,} timestamped words")
    print(f"Assigned {counts['transcript_words']:,} words to segment transcripts")
    print(f"Segments without words: {counts['segments_without_words']:,}")
    print(f"CSV directory: {args.output_dir.expanduser().resolve()}")


if __name__ == "__main__":
    main()
