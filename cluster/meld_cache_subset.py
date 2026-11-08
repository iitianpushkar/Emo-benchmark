"""Select MELD index rows with valid cached Qwen frame entries."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

from cluster.mosei_decoded_cache import cache_available, cache_paths, identity
from scripts.data.prepare_meld import LABELS

SPLITS = ("train", "dev", "test")


def selected_cached_rows(index_dir: Path, cache_dir: Path,
                         protocol: dict) -> tuple[dict[str, list[dict]], dict]:
    manifest = cache_dir.parent / "successful_decodes.csv"
    with manifest.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        required = {"split", "sample_id", "cache_path"}
        if not required <= set(reader.fieldnames or ()):
            raise ValueError(f"{manifest}: missing success columns")
        successes = {}
        for row in reader:
            sample_id = row["sample_id"]
            if not sample_id or row["split"] not in SPLITS or sample_id in successes:
                raise ValueError(f"{manifest}: invalid or duplicate success ID {sample_id}")
            successes[sample_id] = row

    selected = {}
    source_counts = {}
    stale = 0
    seen = set()
    identities = hashlib.sha256()
    for split in SPLITS:
        path = index_dir / f"meld_{split}_index.csv"
        with path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            required = {"sample_id", "split", "video_path", "emotion_id", "video_exists"}
            if not required <= set(reader.fieldnames or ()):
                raise ValueError(f"{path}: missing index columns")
            rows = list(reader)
        source_counts[split] = len(rows)
        chosen = []
        for row in rows:
            sample_id = row["sample_id"]
            if (sample_id in seen or row["split"] != split or
                    row["video_exists"].lower() != "true" or
                    int(row["emotion_id"]) not in range(len(LABELS))):
                raise ValueError(f"{path}: invalid index row {sample_id}")
            seen.add(sample_id)
            success = successes.get(sample_id)
            if success is None:
                continue
            if success["split"] != split:
                raise ValueError(f"{manifest}: split mismatch for {sample_id}")
            expected_path = cache_paths(cache_dir, sample_id)[0].resolve()
            if Path(success["cache_path"]).resolve() != expected_path:
                raise ValueError(f"{manifest}: cache path mismatch for {sample_id}")
            expected = identity(sample_id, Path(row["video_path"]), protocol)
            if cache_available(cache_dir, expected):
                chosen.append(row)
                identities.update(json.dumps(expected, sort_keys=True).encode("utf-8"))
                identities.update(b"\n")
            else:
                stale += 1
        if not chosen:
            raise ValueError(f"No valid cached MELD clips in {split}")
        selected[split] = chosen
    return selected, {"successful_decodes": len(successes), "indexed": source_counts,
                      "selected": {split: len(rows) for split, rows in selected.items()},
                      "stale_success_entries": stale,
                      "cache_identity_sha256": identities.hexdigest()}


def write_selected_indexes(output_dir: Path, rows_by_split: dict[str, list[dict]]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    fields = list(rows_by_split[SPLITS[0]][0])
    for split, rows in rows_by_split.items():
        path = output_dir / f"meld_{split}_index.csv"
        temporary = path.with_suffix(".csv.tmp")
        with temporary.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
        temporary.replace(path)
    path = output_dir.parent / "cached_meld_subset.csv"
    temporary = path.with_suffix(".csv.tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for split in SPLITS:
            writer.writerows(rows_by_split[split])
    temporary.replace(path)


def attach_labels(embedding_path: Path, index_path: Path) -> None:
    import torch

    with index_path.open(newline="", encoding="utf-8") as handle:
        targets = {row["sample_id"]: int(row["emotion_id"]) for row in csv.DictReader(handle)}
    payload = torch.load(embedding_path, map_location="cpu", weights_only=False)
    ids = [str(value) for value in payload["sample_ids"]]
    if len(ids) != len(set(ids)) or not set(ids) <= targets.keys():
        raise ValueError(f"{embedding_path}: sample IDs do not match MELD index")
    payload["labels"] = torch.tensor([targets[sample_id] for sample_id in ids], dtype=torch.long)
    payload["label_names"] = LABELS
    temporary = embedding_path.with_suffix(".pt.tmp")
    torch.save(payload, temporary)
    temporary.replace(embedding_path)
