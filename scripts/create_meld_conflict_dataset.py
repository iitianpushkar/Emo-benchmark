#!/usr/bin/env python3
"""Create deterministic MELD utterance-video conflict pair datasets.

For every utterance in a split, the primary matrix contains one pair for each
video-source emotion. The diagonal keeps the utterance's original matched
video; off-diagonal cells use balanced, cross-dialogue donor videos. A separate
same-class control table replaces the original video with another video having
the same MELD source label.

MELD provides one multimodal emotion label per sample. The generated columns
therefore use ``*_source_emotion`` terminology and do not claim independently
annotated utterance-only or video-only ground truth.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import Counter
from pathlib import Path

import pandas as pd

LABELS = ["anger", "disgust", "fear", "joy", "neutral", "sadness", "surprise"]
REQUIRED_COLUMNS = {
    "sample_id",
    "split",
    "dialogue_id",
    "utterance_id",
    "utterance",
    "emotion",
    "emotion_id",
    "sentiment",
    "video_file",
    "video_path",
    "video_exists",
}

PAIR_COLUMNS = [
    "pair_id",
    "split",
    "pair_type",
    "is_conflict",
    "utterance_source_emotion",
    "video_source_emotion",
    "utterance_emotion_id",
    "video_emotion_id",
    "utterance_sample_id",
    "utterance_dialogue_id",
    "utterance_id",
    "utterance",
    "utterance_source_sentiment",
    "original_video_path",
    "video_sample_id",
    "video_dialogue_id",
    "video_utterance_id",
    "video_source_utterance",
    "video_source_sentiment",
    "video_file",
    "video_path",
    "donor_use_index",
    "donor_pool_size",
    "pairing_seed",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create MELD class-conflict utterance-video pairs.")
    parser.add_argument(
        "--index-dir",
        type=Path,
        default=Path("outputs/indexes"),
        help="Directory containing meld_<split>_index.csv files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/meld_conflict_dataset"),
        help="Directory for generated CSV and metadata files.",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        choices=["dev", "test"],
        default=["dev", "test"],
        help="Evaluation splits to generate.",
    )
    parser.add_argument("--seed", type=int, default=20261001, help="Base pairing seed.")
    parser.add_argument(
        "--skip-same-class-controls",
        action="store_true",
        help="Do not create the separate same-class replacement controls.",
    )
    return parser.parse_args()


def derived_seed(base_seed: int, *parts: str) -> int:
    payload = ":".join([str(base_seed), *parts]).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


def load_index(path: Path, split: str) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"MELD index not found: {path}")

    frame = pd.read_csv(path)
    missing = sorted(REQUIRED_COLUMNS.difference(frame.columns))
    if missing:
        raise ValueError(f"{path} is missing required columns: {missing}")
    if frame["sample_id"].duplicated().any():
        duplicates = frame.loc[frame["sample_id"].duplicated(), "sample_id"].tolist()[:5]
        raise ValueError(f"Duplicate sample_id values in {path}: {duplicates}")
    if set(frame["split"].astype(str)) != {split}:
        raise ValueError(f"{path} contains rows outside split={split!r}")

    frame = frame.copy()
    frame["emotion"] = frame["emotion"].astype(str).str.lower().str.strip()
    unknown = sorted(set(frame["emotion"]).difference(LABELS))
    if unknown:
        raise ValueError(f"Unknown emotion labels in {path}: {unknown}")
    if not frame["video_exists"].astype(bool).all():
        missing_videos = frame.loc[~frame["video_exists"].astype(bool), "sample_id"].tolist()[:5]
        raise ValueError(f"Index contains rows marked with missing videos: {missing_videos}")

    return frame.sort_values("sample_id", kind="stable").reset_index(drop=True)


def balanced_donors(
    sources: list[dict[str, object]],
    donors: list[dict[str, object]],
    seed: int,
) -> list[tuple[dict[str, object], int]]:
    """Assign eligible donors with approximately uniform reuse."""
    if not donors:
        raise ValueError("Cannot assign donors from an empty pool")

    order = list(range(len(donors)))
    random.Random(seed).shuffle(order)
    usage = [0] * len(donors)
    cursor = 0
    assignments: list[tuple[dict[str, object], int]] = []

    for source in sources:
        eligible = [
            index
            for index, donor in enumerate(donors)
            if donor["sample_id"] != source["sample_id"]
            and int(donor["dialogue_id"]) != int(source["dialogue_id"])
        ]
        if not eligible:
            raise ValueError(
                "No eligible cross-dialogue donor for "
                f"source={source['sample_id']} emotion={source['emotion']}"
            )

        minimum_usage = min(usage[index] for index in eligible)
        eligible_set = {index for index in eligible if usage[index] == minimum_usage}

        selected_position = None
        for offset in range(len(order)):
            position = (cursor + offset) % len(order)
            if order[position] in eligible_set:
                selected_position = position
                break
        if selected_position is None:
            raise RuntimeError("Could not select an eligible balanced donor")

        selected_index = order[selected_position]
        usage[selected_index] += 1
        assignments.append((donors[selected_index], usage[selected_index]))
        cursor = (selected_position + 1) % len(order)

    return assignments


def make_pair_row(
    source: dict[str, object],
    donor: dict[str, object],
    pair_type: str,
    donor_use_index: int,
    donor_pool_size: int,
    pairing_seed: int,
) -> dict[str, object]:
    source_emotion = str(source["emotion"])
    video_emotion = str(donor["emotion"])
    pair_id = (
        f"{source['split']}__u-{source['sample_id']}__v-{donor['sample_id']}__{pair_type}"
    )
    return {
        "pair_id": pair_id,
        "split": source["split"],
        "pair_type": pair_type,
        "is_conflict": source_emotion != video_emotion,
        "utterance_source_emotion": source_emotion,
        "video_source_emotion": video_emotion,
        "utterance_emotion_id": int(source["emotion_id"]),
        "video_emotion_id": int(donor["emotion_id"]),
        "utterance_sample_id": source["sample_id"],
        "utterance_dialogue_id": int(source["dialogue_id"]),
        "utterance_id": int(source["utterance_id"]),
        "utterance": source["utterance"],
        "utterance_source_sentiment": source["sentiment"],
        "original_video_path": source["video_path"],
        "video_sample_id": donor["sample_id"],
        "video_dialogue_id": int(donor["dialogue_id"]),
        "video_utterance_id": int(donor["utterance_id"]),
        "video_source_utterance": donor["utterance"],
        "video_source_sentiment": donor["sentiment"],
        "video_file": donor["video_file"],
        "video_path": donor["video_path"],
        "donor_use_index": donor_use_index,
        "donor_pool_size": donor_pool_size,
        "pairing_seed": pairing_seed,
    }


def create_split_pairs(
    frame: pd.DataFrame,
    split: str,
    base_seed: int,
    include_controls: bool,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    records_by_emotion = {
        emotion: frame.loc[frame["emotion"] == emotion].to_dict("records") for emotion in LABELS
    }
    matrix_rows: list[dict[str, object]] = []
    control_rows: list[dict[str, object]] = []

    for utterance_emotion in LABELS:
        sources = records_by_emotion[utterance_emotion]
        for video_emotion in LABELS:
            donors = records_by_emotion[video_emotion]
            cell_seed = derived_seed(base_seed, split, utterance_emotion, video_emotion, "matrix")

            if utterance_emotion == video_emotion:
                for source in sources:
                    matrix_rows.append(
                        make_pair_row(
                            source,
                            source,
                            pair_type="original_aligned",
                            donor_use_index=1,
                            donor_pool_size=len(donors),
                            pairing_seed=cell_seed,
                        )
                    )
                continue

            assignments = balanced_donors(sources, donors, cell_seed)
            for source, (donor, use_index) in zip(sources, assignments):
                matrix_rows.append(
                    make_pair_row(
                        source,
                        donor,
                        pair_type="cross_class_conflict",
                        donor_use_index=use_index,
                        donor_pool_size=len(donors),
                        pairing_seed=cell_seed,
                    )
                )

        if include_controls:
            donors = records_by_emotion[utterance_emotion]
            control_seed = derived_seed(base_seed, split, utterance_emotion, "same_class_control")
            assignments = balanced_donors(sources, donors, control_seed)
            for source, (donor, use_index) in zip(sources, assignments):
                control_rows.append(
                    make_pair_row(
                        source,
                        donor,
                        pair_type="same_class_replacement",
                        donor_use_index=use_index,
                        donor_pool_size=len(donors),
                        pairing_seed=control_seed,
                    )
                )

    matrix = pd.DataFrame(matrix_rows, columns=PAIR_COLUMNS)
    controls = pd.DataFrame(control_rows, columns=PAIR_COLUMNS)
    return matrix, controls


def validate_pairs(
    source: pd.DataFrame,
    matrix: pd.DataFrame,
    controls: pd.DataFrame,
    include_controls: bool,
) -> None:
    expected_matrix_rows = len(source) * len(LABELS)
    if len(matrix) != expected_matrix_rows:
        raise AssertionError(f"Expected {expected_matrix_rows} matrix rows, found {len(matrix)}")
    if matrix["pair_id"].duplicated().any():
        raise AssertionError("Matrix contains duplicate pair_id values")

    expected_source_counts = source["emotion"].value_counts().reindex(LABELS, fill_value=0)
    cell_counts = matrix.groupby(
        ["utterance_source_emotion", "video_source_emotion"], observed=False
    ).size()
    for utterance_emotion in LABELS:
        for video_emotion in LABELS:
            observed = int(cell_counts.get((utterance_emotion, video_emotion), 0))
            expected = int(expected_source_counts[utterance_emotion])
            if observed != expected:
                raise AssertionError(
                    f"Cell ({utterance_emotion}, {video_emotion}) has {observed} rows; expected {expected}"
                )

    original = matrix["pair_type"] == "original_aligned"
    conflict = matrix["pair_type"] == "cross_class_conflict"
    if not (matrix.loc[original, "utterance_sample_id"] == matrix.loc[original, "video_sample_id"]).all():
        raise AssertionError("Original aligned rows must preserve the matched video")
    if matrix.loc[original, "is_conflict"].any():
        raise AssertionError("Original aligned rows cannot be conflicts")
    if not matrix.loc[conflict, "is_conflict"].all():
        raise AssertionError("Cross-class rows must be marked as conflicts")
    if not (
        matrix.loc[conflict, "utterance_dialogue_id"]
        != matrix.loc[conflict, "video_dialogue_id"]
    ).all():
        raise AssertionError("Cross-class donors must come from another dialogue")

    per_source_video_classes = matrix.groupby("utterance_sample_id")["video_source_emotion"].nunique()
    if not (per_source_video_classes == len(LABELS)).all():
        raise AssertionError("Every utterance must be paired with all seven video-source classes")

    if include_controls:
        if len(controls) != len(source):
            raise AssertionError(f"Expected {len(source)} control rows, found {len(controls)}")
        if controls["pair_id"].duplicated().any():
            raise AssertionError("Controls contain duplicate pair_id values")
        if controls["is_conflict"].any():
            raise AssertionError("Same-class controls cannot be marked as conflicts")
        if not (
            controls["utterance_source_emotion"] == controls["video_source_emotion"]
        ).all():
            raise AssertionError("Same-class controls must preserve the source emotion")
        if not (controls["utterance_sample_id"] != controls["video_sample_id"]).all():
            raise AssertionError("Same-class controls must replace the original video")
        if not (controls["utterance_dialogue_id"] != controls["video_dialogue_id"]).all():
            raise AssertionError("Same-class controls must use another dialogue")


def summarize_pairs(matrix: pd.DataFrame, controls: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for dataset_name, frame in [("matrix", matrix), ("same_class_controls", controls)]:
        if frame.empty:
            continue
        grouped = frame.groupby(
            ["split", "pair_type", "utterance_source_emotion", "video_source_emotion"],
            sort=False,
        )
        for keys, group in grouped:
            split, pair_type, utterance_emotion, video_emotion = keys
            donor_counts = Counter(group["video_sample_id"])
            rows.append(
                {
                    "split": split,
                    "dataset": dataset_name,
                    "pair_type": pair_type,
                    "utterance_source_emotion": utterance_emotion,
                    "video_source_emotion": video_emotion,
                    "pair_count": len(group),
                    "unique_utterances": group["utterance_sample_id"].nunique(),
                    "unique_videos": group["video_sample_id"].nunique(),
                    "minimum_donor_reuse": min(donor_counts.values()),
                    "maximum_donor_reuse": max(donor_counts.values()),
                }
            )
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    index_dir = args.index_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    include_controls = not args.skip_same_class_controls

    all_summaries = []
    split_metadata: dict[str, object] = {}

    for split in args.splits:
        source_path = index_dir / f"meld_{split}_index.csv"
        source = load_index(source_path, split)
        matrix, controls = create_split_pairs(source, split, args.seed, include_controls)
        validate_pairs(source, matrix, controls, include_controls)

        matrix_path = output_dir / f"meld_{split}_conflict_matrix.csv"
        matrix.to_csv(matrix_path, index=False)
        print(f"[{split}] matrix rows: {len(matrix):,} -> {matrix_path}")

        controls_path = None
        if include_controls:
            controls_path = output_dir / f"meld_{split}_same_class_controls.csv"
            controls.to_csv(controls_path, index=False)
            print(f"[{split}] same-class controls: {len(controls):,} -> {controls_path}")

        summary = summarize_pairs(matrix, controls)
        all_summaries.append(summary)
        split_metadata[split] = {
            "source_index": str(source_path),
            "source_rows": len(source),
            "matrix_rows": len(matrix),
            "conflict_rows": int(matrix["is_conflict"].sum()),
            "original_aligned_rows": int((matrix["pair_type"] == "original_aligned").sum()),
            "same_class_control_rows": len(controls),
            "matrix_csv": str(matrix_path),
            "same_class_controls_csv": str(controls_path) if controls_path else None,
        }

    pair_counts = pd.concat(all_summaries, ignore_index=True)
    counts_path = output_dir / "meld_conflict_pair_counts.csv"
    pair_counts.to_csv(counts_path, index=False)

    metadata = {
        "dataset": "MELD utterance-video class conflict matrix",
        "labels": LABELS,
        "base_seed": args.seed,
        "splits": split_metadata,
        "protocol": {
            "matrix": (
                "For each utterance, retain its original video for the matching source-emotion cell and "
                "assign one balanced cross-dialogue donor video from every other source-emotion class."
            ),
            "same_class_controls": (
                "For each utterance, assign one balanced cross-dialogue donor video with the same MELD "
                "source-emotion label."
            ),
            "donor_reuse": "Balanced within each ordered utterance-class/video-class cell.",
        },
        "label_limitation": (
            "MELD labels describe the original multimodal sample. The utterance_source_emotion and "
            "video_source_emotion fields are inherited source labels, not independent modality-only annotations."
        ),
    }
    metadata_path = output_dir / "meld_conflict_dataset_metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")

    print(f"Pair-count summary -> {counts_path}")
    print(f"Metadata -> {metadata_path}")


if __name__ == "__main__":
    main()
