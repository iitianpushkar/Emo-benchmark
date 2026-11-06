"""Select single-label MOSEI clips with valid predecoded frame cache entries."""

from __future__ import annotations

import csv
import time
from collections import Counter
from pathlib import Path

from cluster.mosei_decoded_cache import cache_available, cache_paths, identity

SPLITS = ("train", "valid", "test")


def resolve_cache_frames(path: Path) -> Path:
    path = path.expanduser().resolve()
    if (path / "successful_decodes.csv").is_file() and (path / "frames").is_dir():
        return path / "frames"
    if path.name == "frames" and (path.parent / "successful_decodes.csv").is_file():
        return path
    raise ValueError(f"Expected a decoded cache root or its frames directory: {path}")


def cached_single_label_rows(single_label_csv: Path, cache_dir: Path,
                             raw_dir: Path, protocol: dict) -> tuple[list[str], list[dict], dict]:
    started = time.monotonic()
    manifest = cache_dir.parent / "successful_decodes.csv"
    if not manifest.is_file():
        raise FileNotFoundError(f"Decode success manifest not found: {manifest}")
    print(f"Reading successful decode IDs from {manifest}", flush=True)
    successful = {}
    with manifest.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        required = {"split", "sample_id", "cache_path"}
        if not required <= set(reader.fieldnames or ()):
            raise ValueError(f"{manifest}: missing {sorted(required - set(reader.fieldnames or ()))}")
        for row in reader:
            sample_id, split = row["sample_id"], row["split"]
            if not sample_id or split not in SPLITS or sample_id in successful:
                raise ValueError(f"{manifest}: invalid or duplicate success: {sample_id}, {split}")
            successful[sample_id] = row
    print(f"Loaded {len(successful)} successful decode IDs in "
          f"{time.monotonic() - started:.1f}s; checking labels and cached files", flush=True)

    counts = Counter()
    seen = set()
    selected = []
    with single_label_csv.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        required = {"sample_id", "split", "video_id", "clip_id", "emotion", "emotion_id", "label_status"}
        if not required <= set(reader.fieldnames or ()):
            raise ValueError(f"{single_label_csv}: missing {sorted(required - set(reader.fieldnames or ()))}")
        fields = list(reader.fieldnames or ())
        for checked, row in enumerate(reader, start=1):
            sample_id, split = row["sample_id"], row["split"]
            if sample_id in seen or split not in SPLITS:
                raise ValueError(f"{single_label_csv}: duplicate ID or invalid split: {sample_id}, {split}")
            seen.add(sample_id)
            counts["single_label_rows"] += 1
            if row["label_status"] not in {"unique_max", "all_zero"}:
                raise ValueError(f"{single_label_csv}: invalid label status for {sample_id}")
            success = successful.get(sample_id)
            if success is None:
                if checked % 500 == 0:
                    print(f"Cache selection: {checked} labels checked; {len(selected)} selected; "
                          f"{counts['stale_success_entries']} stale; "
                          f"{time.monotonic() - started:.1f}s elapsed", flush=True)
                continue
            if split != success["split"]:
                raise ValueError(f"{manifest}: split mismatch for {sample_id}")
            video_path = raw_dir / row["video_id"] / f"{row['clip_id']}.mp4"
            expected = identity(sample_id, video_path, protocol)
            if Path(success["cache_path"]).resolve() != cache_paths(cache_dir, sample_id)[0].resolve():
                raise ValueError(f"{manifest}: cache path mismatch for {sample_id}")
            if not cache_available(cache_dir, expected):
                counts["stale_success_entries"] += 1
            else:
                selected.append(row)
                counts[split] += 1
            if checked % 500 == 0:
                print(f"Cache selection: {checked} labels checked; {len(selected)} selected; "
                      f"{counts['stale_success_entries']} stale; "
                      f"{time.monotonic() - started:.1f}s elapsed", flush=True)

    if any(counts[split] == 0 for split in SPLITS):
        raise ValueError("Cache and single-label CSV must overlap in every split")
    summary = {"successful_decodes": len(successful),
               "single_label_rows": counts["single_label_rows"],
               "selected": {split: counts[split] for split in SPLITS},
               "stale_success_entries": counts["stale_success_entries"]}
    print(f"Cache selection complete: {counts['single_label_rows']} labels checked; "
          f"{len(selected)} selected in {time.monotonic() - started:.1f}s", flush=True)
    return fields, selected, summary


def write_subset(path: Path, fields: list[str], rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    try:
        with temporary.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)
