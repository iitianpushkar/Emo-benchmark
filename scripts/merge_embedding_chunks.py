#!/usr/bin/env python3
"""Merge chunked embedding .pt files produced by extraction scripts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Merge extracted embedding chunk .pt files.")
    parser.add_argument("--chunks-dir", type=Path, required=True, help="Directory containing chunk .pt files.")
    parser.add_argument("--output-pt", type=Path, required=True, help="Merged output .pt path.")
    parser.add_argument("--pattern", default="*.pt", help="Glob pattern for chunk files.")
    return parser.parse_args()


def load_chunk(path: Path) -> dict[str, Any]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    required = ["embeddings", "labels"]
    missing = [key for key in required if key not in payload]
    if missing:
        raise ValueError(f"{path} missing keys: {missing}")
    return payload


def main() -> None:
    args = parse_args()
    chunk_paths = sorted(args.chunks_dir.glob(args.pattern))
    if not chunk_paths:
        raise FileNotFoundError(f"No chunk files found in {args.chunks_dir} with pattern {args.pattern}")

    embeddings = []
    labels = []
    sample_ids = []
    metadata = []
    errors = []
    label_names = None
    configs = []

    for path in chunk_paths:
        payload = load_chunk(path)
        x = payload["embeddings"]
        y = payload["labels"]
        if x.shape[0] != y.shape[0]:
            raise ValueError(f"{path} embeddings/labels count mismatch: {x.shape[0]} vs {y.shape[0]}")

        embeddings.append(x.cpu())
        labels.append(y.cpu().long())
        sample_ids.extend(payload.get("sample_ids", [f"{path.stem}_{i}" for i in range(x.shape[0])]))
        metadata.extend(payload.get("metadata", [{} for _ in range(x.shape[0])]))
        errors.extend(payload.get("errors", []))
        configs.append({"chunk": str(path), "config": payload.get("config", {})})
        if label_names is None:
            label_names = payload.get("label_names")

        print(f"Loaded {path}: {tuple(x.shape)}, errors={len(payload.get('errors', []))}")

    merged = {
        "embeddings": torch.cat(embeddings, dim=0),
        "labels": torch.cat(labels, dim=0),
        "label_names": label_names,
        "sample_ids": sample_ids,
        "metadata": metadata,
        "errors": errors,
        "config": {"merged_from": configs},
    }

    args.output_pt.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = args.output_pt.with_suffix(args.output_pt.suffix + ".tmp")
    torch.save(merged, tmp_path)
    tmp_path.replace(args.output_pt)

    print(f"Saved merged embeddings: {args.output_pt}")
    print(f"Embedding tensor: {tuple(merged['embeddings'].shape)}")
    print(f"Labels: {tuple(merged['labels'].shape)}")
    print(f"Total errors from chunks: {len(errors)}")
    if errors:
        error_path = args.output_pt.with_suffix(".errors.json")
        error_path.write_text(json.dumps(errors, indent=2), encoding="utf-8")
        print(f"Saved merged errors: {error_path}")


if __name__ == "__main__":
    main()
