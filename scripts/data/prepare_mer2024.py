#!/usr/bin/env python3
"""Build MELD-compatible indexes for the MER2024 labeled release."""

from __future__ import annotations

import argparse
import html
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


LABELS = ["angry", "happy", "neutral", "sad", "surprise", "worried"]
LABEL_ALIASES = {"anger": "angry", "sadness": "sad", "worry": "worried"}
VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mer-root", type=Path, default=Path("dataset/MER2024"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/indexes"))
    parser.add_argument("--dev-ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--include-missing-transcripts",
        action="store_true",
        help="Keep the officially labeled clips whose English and Chinese transcripts are both empty.",
    )
    return parser.parse_args()


def clean_text(value: object) -> str:
    return "" if pd.isna(value) else " ".join(html.unescape(str(value)).split())


def prepare_indexes(
    mer_root: Path,
    dev_ratio: float,
    seed: int,
    expected_count: int = 5030,
    require_transcript: bool = True,
) -> dict[str, pd.DataFrame]:
    labels_path = mer_root / "label-disdim.csv"
    transcripts_path = mer_root / "label-transcription.csv"
    if not labels_path.is_file() or not transcripts_path.is_file():
        raise FileNotFoundError("Missing label-disdim.csv or label-transcription.csv")

    labels = pd.read_csv(labels_path)
    transcripts = pd.read_csv(transcripts_path)
    required_labels = {"name", "discrete"}
    required_transcripts = {"name", "chinese", "english"}
    if not required_labels.issubset(labels.columns):
        raise ValueError(f"Label columns are {labels.columns.tolist()}, expected {sorted(required_labels)}")
    if not required_transcripts.issubset(transcripts.columns):
        raise ValueError(
            f"Transcription columns are {transcripts.columns.tolist()}, expected {sorted(required_transcripts)}"
        )

    labels["name"] = labels["name"].astype(str).str.strip()
    transcripts["name"] = transcripts["name"].astype(str).str.strip()
    if labels["name"].duplicated().any() or transcripts["name"].duplicated().any():
        raise ValueError("MER2024 metadata contains duplicate sample names")

    metadata = labels.merge(transcripts, on="name", how="inner", validate="one_to_one")
    video_paths = [path for path in mer_root.rglob("*") if path.suffix.lower() in VIDEO_EXTENSIONS]
    video_map = {path.stem: path.resolve() for path in video_paths}
    if len(video_map) != len(video_paths):
        raise ValueError("Video filenames are not unique by stem")

    out = pd.DataFrame()
    out["sample_id"] = "mer2024_" + metadata["name"]
    out["utterance_id"] = metadata["name"]
    out["utterance_english"] = metadata["english"].map(clean_text)
    out["utterance_chinese"] = metadata["chinese"].map(clean_text)
    out["utterance"] = out["utterance_english"].where(
        out["utterance_english"].ne(""), out["utterance_chinese"]
    )
    out["has_transcript"] = out["utterance"].ne("")
    out["emotion"] = metadata["discrete"].astype(str).str.lower().str.strip().replace(LABEL_ALIASES)
    unknown = sorted(set(out["emotion"]) - set(LABELS))
    if unknown:
        raise ValueError(f"Unknown MER2024 labels: {unknown}")
    out["emotion_id"] = out["emotion"].map({label: index for index, label in enumerate(LABELS)})
    if "valence" in metadata:
        out["valence"] = pd.to_numeric(metadata["valence"], errors="coerce")
    out["video_file"] = metadata["name"].map(lambda name: video_map.get(name, Path(f"{name}.mp4")).name)
    out["video_path"] = metadata["name"].map(lambda name: str(video_map[name]) if name in video_map else "")
    out["video_exists"] = metadata["name"].isin(video_map)

    if len(out) != expected_count or not out["video_exists"].all():
        missing = out.loc[~out["video_exists"], "utterance_id"].head(10).tolist()
        raise RuntimeError(
            f"Expected {expected_count:,} aligned samples, found {len(out):,}; missing videos: {missing}"
        )

    if require_transcript:
        out = out[out["has_transcript"]].copy()

    train_indices, dev_indices = train_test_split(
        out.index,
        test_size=dev_ratio,
        random_state=seed,
        stratify=out["emotion"],
    )
    out["split"] = "train"
    out.loc[dev_indices, "split"] = "dev"
    out["split_source"] = f"stratified_dev_{dev_ratio:g}_seed_{seed}"
    columns = ["sample_id", "split", *[column for column in out.columns if column not in {"sample_id", "split"}]]
    out = out[columns].sort_values(["split", "sample_id"]).reset_index(drop=True)
    return {
        "train": out[out["split"] == "train"].reset_index(drop=True),
        "dev": out[out["split"] == "dev"].reset_index(drop=True),
        "all": out,
    }


def main() -> None:
    args = parse_args()
    indexes = prepare_indexes(
        args.mer_root.expanduser().resolve(),
        args.dev_ratio,
        args.seed,
        require_transcript=not args.include_missing_transcripts,
    )
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    for split, frame in indexes.items():
        path = output_dir / f"mer2024_{split}_index.csv"
        frame.to_csv(path, index=False)
        print(f"[{split}] {len(frame):,} rows -> {path}")
        print(frame["emotion"].value_counts().reindex(LABELS, fill_value=0).to_string())


if __name__ == "__main__":
    main()
