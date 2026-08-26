#!/usr/bin/env python3
"""Build clean MELD split indexes with verified video paths.

This script does not copy or modify videos. It reads MELD CSV metadata,
constructs the expected video path for each utterance, checks whether the file
exists, and writes a compact index CSV for downstream embedding extraction.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import pandas as pd

LABELS = ["anger", "disgust", "fear", "joy", "neutral", "sadness", "surprise"]

SPLIT_LAYOUTS = {
    "train": {
        "csv": ["train/train_sent_emo.csv"],
        "video_dirs": ["train/train_splits", "train_splits"],
    },
    "dev": {
        "csv": ["dev_sent_emo.csv", "dev/dev_sent_emo.csv"],
        "video_dirs": ["dev_splits_complete", "dev/dev_splits_complete", "dev_splits", "dev/dev_splits"],
    },
    "test": {
        "csv": ["test_sent_emo.csv", "test/test_sent_emo.csv"],
        "video_dirs": [
            "output_repeated_splits_test",
            "test/output_repeated_splits_test",
            "test_splits",
            "test/test_splits",
        ],
    },
}

REQUIRED_COLUMNS = ["Utterance", "Emotion", "Dialogue_ID", "Utterance_ID"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare MELD metadata indexes.")
    parser.add_argument(
        "--meld-root",
        type=Path,
        default=Path("dataset/MELD.Raw"),
        help="Path to MELD.Raw directory.",
    )
    parser.add_argument(
        "--split",
        choices=["train", "dev", "test", "all"],
        default="all",
        help="Dataset split to prepare.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/indexes"),
        help="Directory where index CSVs will be written.",
    )
    parser.add_argument(
        "--include-missing",
        action="store_true",
        help="Keep rows whose expected video file is missing.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional maximum number of existing-video rows per split.",
    )
    return parser.parse_args()


def first_existing(root: Path, candidates: Iterable[str]) -> Path | None:
    for rel_path in candidates:
        path = root / rel_path
        if path.exists():
            return path
    return None


def clean_text(value: object) -> str:
    """Fix common Windows-1252 mojibake while preserving the original wording."""
    text = "" if pd.isna(value) else str(value)
    replacements = {
        "\u0092": "'",
        "\u0091": "'",
        "\u0093": '"',
        "\u0094": '"',
        "\u0096": "-",
        "\u0097": "-",
        "\u00a0": " ",
    }
    for bad, good in replacements.items():
        text = text.replace(bad, good)
    return " ".join(text.split())


def prepare_split(meld_root: Path, split: str, include_missing: bool, limit: int | None) -> pd.DataFrame:
    layout = SPLIT_LAYOUTS[split]
    csv_path = first_existing(meld_root, layout["csv"])
    video_dir = first_existing(meld_root, layout["video_dirs"])

    if csv_path is None:
        tried = [str(meld_root / item) for item in layout["csv"]]
        raise FileNotFoundError(f"Could not find {split} CSV. Tried: {tried}")
    if video_dir is None:
        tried = [str(meld_root / item) for item in layout["video_dirs"]]
        raise FileNotFoundError(f"Could not find {split} video directory. Tried: {tried}")

    df = pd.read_csv(csv_path)
    missing_columns = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing_columns:
        raise ValueError(f"{csv_path} is missing columns: {missing_columns}")

    out = pd.DataFrame()
    out["sample_id"] = df.apply(
        lambda row: f"{split}_dia{int(row['Dialogue_ID'])}_utt{int(row['Utterance_ID'])}", axis=1
    )
    out["split"] = split
    out["dialogue_id"] = df["Dialogue_ID"].astype(int)
    out["utterance_id"] = df["Utterance_ID"].astype(int)
    out["utterance"] = df["Utterance"].map(clean_text)
    out["emotion"] = df["Emotion"].astype(str).str.lower().str.strip()
    out["emotion_id"] = out["emotion"].map({label: idx for idx, label in enumerate(LABELS)})
    out["sentiment"] = df["Sentiment"].astype(str).str.lower().str.strip() if "Sentiment" in df else ""
    out["video_file"] = out.apply(lambda row: f"dia{row['dialogue_id']}_utt{row['utterance_id']}.mp4", axis=1)
    out["video_path"] = out["video_file"].map(lambda name: str((video_dir / name).resolve()))
    out["video_exists"] = out["video_path"].map(lambda path: Path(path).exists())

    if out["emotion_id"].isna().any():
        unknown = sorted(out.loc[out["emotion_id"].isna(), "emotion"].unique())
        raise ValueError(f"Unknown emotion labels in {csv_path}: {unknown}")

    out["emotion_id"] = out["emotion_id"].astype(int)

    if not include_missing:
        out = out[out["video_exists"]].copy()

    if limit is not None:
        out = out.head(limit).copy()

    return out.reset_index(drop=True)


def main() -> None:
    args = parse_args()
    meld_root = args.meld_root.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if not meld_root.exists():
        raise FileNotFoundError(f"MELD root not found: {meld_root}")

    splits = ["train", "dev", "test"] if args.split == "all" else [args.split]
    prepared = []

    for split in splits:
        split_df = prepare_split(meld_root, split, args.include_missing, args.limit)
        prepared.append(split_df)

        output_path = output_dir / f"meld_{split}_index.csv"
        split_df.to_csv(output_path, index=False)

        label_counts = split_df["emotion"].value_counts().reindex(LABELS, fill_value=0)
        print(f"[{split}] rows written: {len(split_df)} -> {output_path}")
        print(label_counts.to_string())
        print()

    if len(prepared) > 1:
        all_df = pd.concat(prepared, ignore_index=True)
        output_path = output_dir / "meld_all_index.csv"
        all_df.to_csv(output_path, index=False)
        print(f"[all] rows written: {len(all_df)} -> {output_path}")


if __name__ == "__main__":
    main()
