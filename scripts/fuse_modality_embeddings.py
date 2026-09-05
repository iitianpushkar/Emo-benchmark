#!/usr/bin/env python3
"""Fuse video-only and text-only Qwen LM-space embeddings with alpha weighting."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

try:
    import torch
except ModuleNotFoundError:
    torch = None  # type: ignore[assignment]


SPLITS = ["train", "dev", "test"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create alpha-weighted fused modality embeddings.")
    parser.add_argument("--video-train-pt", type=Path, required=True, help="Video-only train embeddings.")
    parser.add_argument("--text-train-pt", type=Path, required=True, help="Text-only train embeddings.")
    parser.add_argument("--video-dev-pt", type=Path, required=True, help="Video-only dev embeddings.")
    parser.add_argument("--text-dev-pt", type=Path, required=True, help="Text-only dev embeddings.")
    parser.add_argument("--video-test-pt", type=Path, default=None, help="Optional video-only test embeddings.")
    parser.add_argument("--text-test-pt", type=Path, default=None, help="Optional text-only test embeddings.")
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory for fused .pt files.")
    parser.add_argument("--alpha", type=float, required=True, help="Video weight. Text weight is 1 - alpha.")
    parser.add_argument(
        "--no-normalize",
        action="store_true",
        help="Fuse raw embeddings instead of standardizing each modality with train-set statistics.",
    )
    return parser.parse_args()


def load_payload(path: Path) -> dict[str, Any]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    required = ["embeddings", "labels", "sample_ids"]
    missing = [key for key in required if key not in payload]
    if missing:
        raise ValueError(f"{path} missing keys: {missing}")
    return payload


def payload_by_sample_id(payload: dict[str, Any], path: Path) -> dict[str, int]:
    sample_ids = [str(sample_id) for sample_id in payload["sample_ids"]]
    if len(sample_ids) != len(set(sample_ids)):
        raise ValueError(f"{path} contains duplicate sample_ids")
    if payload["embeddings"].shape[0] != len(sample_ids):
        raise ValueError(f"{path} embeddings rows do not match sample_ids length")
    if payload["labels"].shape[0] != len(sample_ids):
        raise ValueError(f"{path} labels rows do not match sample_ids length")
    return {sample_id: idx for idx, sample_id in enumerate(sample_ids)}


def align_text_to_video(
    video_payload: dict[str, Any],
    text_payload: dict[str, Any],
    video_path: Path,
    text_path: Path,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, list[str], list[dict[str, Any]]]:
    video_index = payload_by_sample_id(video_payload, video_path)
    text_index = payload_by_sample_id(text_payload, text_path)

    video_ids = [str(sample_id) for sample_id in video_payload["sample_ids"]]
    missing_from_text = [sample_id for sample_id in video_ids if sample_id not in text_index]
    extra_text = [sample_id for sample_id in text_index if sample_id not in video_index]
    if missing_from_text or extra_text:
        raise ValueError(
            f"Sample id mismatch between {video_path} and {text_path}: "
            f"{len(missing_from_text)} missing from text, {len(extra_text)} extra in text"
        )

    text_order = [text_index[sample_id] for sample_id in video_ids]
    video_x = video_payload["embeddings"].float()
    text_x = text_payload["embeddings"][text_order].float()
    labels = video_payload["labels"].long()
    text_labels = text_payload["labels"][text_order].long()

    if video_x.ndim != 2 or text_x.ndim != 2:
        raise ValueError("Expected video and text embeddings to be 2D tensors")
    if video_x.shape != text_x.shape:
        raise ValueError(f"Embedding shape mismatch: video {tuple(video_x.shape)} vs text {tuple(text_x.shape)}")
    if not torch.equal(labels, text_labels):
        raise ValueError(f"Label mismatch after sample_id alignment between {video_path} and {text_path}")

    video_meta = video_payload.get("metadata", [{} for _ in video_ids])
    text_meta = text_payload.get("metadata", [{} for _ in video_ids])
    metadata = []
    for i, sample_id in enumerate(video_ids):
        row = dict(video_meta[i] if i < len(video_meta) and isinstance(video_meta[i], dict) else {})
        row["sample_id"] = sample_id
        row["fusion_source"] = "alpha_video_plus_text"
        row["text_metadata"] = text_meta[text_order[i]] if text_order[i] < len(text_meta) else {}
        metadata.append(row)

    return video_x, text_x, labels, video_ids, metadata


def fit_standardizer(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    mean = x.mean(dim=0, keepdim=True)
    std = x.std(dim=0, keepdim=True).clamp_min(1e-6)
    return mean, std


def transform(x: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    return (x - mean) / std


def save_fused_payload(
    path: Path,
    fused: torch.Tensor,
    labels: torch.Tensor,
    sample_ids: list[str],
    metadata: list[dict[str, Any]],
    label_names: list[str],
    config: dict[str, Any],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "embeddings": fused,
        "labels": labels,
        "label_names": label_names,
        "sample_ids": sample_ids,
        "metadata": metadata,
        "config": config,
    }
    torch.save(payload, path)


def main() -> None:
    args = parse_args()
    if torch is None:
        raise ModuleNotFoundError("fuse_modality_embeddings.py requires torch. Install dependencies from requirements.txt.")
    if not 0.0 <= args.alpha <= 1.0:
        raise ValueError("--alpha must be between 0 and 1")
    if (args.video_test_pt is None) != (args.text_test_pt is None):
        raise ValueError("Provide both --video-test-pt and --text-test-pt, or neither")

    pairs: dict[str, tuple[Path, Path]] = {
        "train": (args.video_train_pt, args.text_train_pt),
        "dev": (args.video_dev_pt, args.text_dev_pt),
    }
    if args.video_test_pt is not None and args.text_test_pt is not None:
        pairs["test"] = (args.video_test_pt, args.text_test_pt)

    aligned: dict[str, tuple[torch.Tensor, torch.Tensor, torch.Tensor, list[str], list[dict[str, Any]], dict[str, Any]]] = {}
    for split, (video_path, text_path) in pairs.items():
        video_payload = load_payload(video_path)
        text_payload = load_payload(text_path)
        video_x, text_x, labels, sample_ids, metadata = align_text_to_video(
            video_payload,
            text_payload,
            video_path,
            text_path,
        )
        label_names = video_payload.get("label_names", text_payload.get("label_names", []))
        aligned[split] = (video_x, text_x, labels, sample_ids, metadata, {"label_names": label_names})

    video_mean, video_std = fit_standardizer(aligned["train"][0])
    text_mean, text_std = fit_standardizer(aligned["train"][1])

    args.output_dir.mkdir(parents=True, exist_ok=True)
    for split, (video_x, text_x, labels, sample_ids, metadata, extras) in aligned.items():
        if args.no_normalize:
            video_part = video_x
            text_part = text_x
        else:
            video_part = transform(video_x, video_mean, video_std)
            text_part = transform(text_x, text_mean, text_std)

        fused = args.alpha * video_part + (1.0 - args.alpha) * text_part
        config = {
            "fusion": "alpha_video_plus_text",
            "alpha_video": args.alpha,
            "beta_text": 1.0 - args.alpha,
            "normalized_before_fusion": not args.no_normalize,
            "video_path": str(pairs[split][0]),
            "text_path": str(pairs[split][1]),
        }
        save_fused_payload(
            args.output_dir / f"meld_{split}.pt",
            fused=fused,
            labels=labels,
            sample_ids=sample_ids,
            metadata=metadata,
            label_names=extras["label_names"],
            config=config,
        )
        print(f"[{split}] saved {args.output_dir / f'meld_{split}.pt'} with shape {tuple(fused.shape)}")


if __name__ == "__main__":
    main()
