#!/usr/bin/env python3
"""Keep seven-class MOSEI rows with successful baseline Qwen embeddings."""

from __future__ import annotations

import argparse
import csv
from collections import Counter
from pathlib import Path


def filter_rows(single_label_csv: Path, successful_ids_csv: Path, output_csv: Path) -> dict:
    if output_csv.resolve() in {single_label_csv.resolve(), successful_ids_csv.resolve()}:
        raise ValueError("Output CSV must differ from both input CSVs")

    successful = {}
    with successful_ids_csv.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if not {"sample_id", "split"} <= set(reader.fieldnames or ()):
            raise ValueError("Successful-ID CSV requires sample_id and split columns")
        for row in reader:
            sample_id, split = row["sample_id"], row["split"]
            if not sample_id or split not in {"train", "valid", "test"}:
                raise ValueError(f"Invalid successful-ID row: {row}")
            if sample_id in successful:
                raise ValueError(f"Duplicate successful sample ID: {sample_id}")
            successful[sample_id] = split

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_csv.with_name(output_csv.name + ".tmp")
    written = Counter()
    seen = set()
    matched = set()
    try:
        with single_label_csv.open(newline="", encoding="utf-8") as source, temporary.open(
            "w", newline="", encoding="utf-8"
        ) as destination:
            reader = csv.DictReader(source)
            required = {"sample_id", "split", "emotion", "emotion_id", "label_status"}
            if not required <= set(reader.fieldnames or ()):
                raise ValueError(f"Single-label CSV missing {sorted(required - set(reader.fieldnames or ()))}")
            writer = csv.DictWriter(destination, fieldnames=reader.fieldnames)
            writer.writeheader()
            for row in reader:
                sample_id = row["sample_id"]
                if sample_id in seen:
                    raise ValueError(f"Duplicate single-label sample ID: {sample_id}")
                seen.add(sample_id)
                if row["label_status"] not in {"unique_max", "all_zero"}:
                    raise ValueError(f"Unexpected label status for {sample_id}: {row['label_status']}")
                if sample_id not in successful:
                    continue
                if successful[sample_id] != row["split"]:
                    raise ValueError(f"Split mismatch for {sample_id}")
                writer.writerow(row)
                written[row["split"]] += 1
                matched.add(sample_id)
        temporary.replace(output_csv)
    finally:
        temporary.unlink(missing_ok=True)

    return {"successful_embeddings": len(successful), "eligible_and_embedded": len(matched),
            "written_by_split": {split: written[split] for split in ("train", "valid", "test")},
            "successful_ids_without_eligible_label": len(successful) - len(matched)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--single-label-csv", type=Path,
                        default=Path("dataset/CMU-MOSEI/csv/mosei_single_label.csv"))
    parser.add_argument("--successful-ids-csv", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path,
                        default=Path("dataset/CMU-MOSEI/csv/mosei_single_label_qwen7b_embedded.csv"))
    args = parser.parse_args()
    summary = filter_rows(args.single_label_csv, args.successful_ids_csv, args.output_csv)
    print(summary)
    print(f"Filtered single-label CSV: {args.output_csv}")


if __name__ == "__main__":
    main()
