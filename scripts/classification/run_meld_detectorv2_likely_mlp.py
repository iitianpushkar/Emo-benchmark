#!/usr/bin/env python3
"""Retrain a MELD MLP on Detectorv2 likely-speaker embeddings and compare baseline."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import zipfile
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score

from scripts.classification.train_mlp import DEFAULT_LABELS

REPO = Path(__file__).resolve().parents[2]
SPLITS = ("train", "dev", "test")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, required=True,
                        help="Directory containing detectorv2_prompt/meld_*.pt, meld_*.pt, or visual_prompt_meld_*.zip")
    parser.add_argument("--baseline-dir", type=Path,
                        default=REPO / "experiments/baselines/qwen2_5_vl_3b_shared/outputs")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default="auto")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def locate_embedding(results_dir: Path, split: str, staging_dir: Path) -> Path:
    for candidate in (results_dir / f"meld_{split}.pt",
                      results_dir / "detectorv2_prompt" / f"meld_{split}.pt"):
        if candidate.is_file():
            return candidate
    archive = results_dir / f"visual_prompt_meld_{split}.zip"
    if not archive.is_file():
        raise FileNotFoundError(f"Missing meld_{split}.pt or {archive}")
    member = f"detectorv2_prompt/meld_{split}.pt"
    staging_dir.mkdir(parents=True, exist_ok=True)
    target = staging_dir / f"meld_{split}.pt"
    with zipfile.ZipFile(archive) as zf:
        info = zf.getinfo(member)
        if not target.exists() or target.stat().st_size != info.file_size:
            with zf.open(info) as src, target.open("wb") as dst:
                while chunk := src.read(1024 * 1024):
                    dst.write(chunk)
    return target


def filter_likely(path: Path, split: str, target: Path) -> dict:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    ids = [str(x) for x in payload["sample_ids"]]
    metadata = payload["metadata"]
    labels = payload["labels"].long()
    embeddings = payload["embeddings"]
    names = payload.get("label_names")
    if names != DEFAULT_LABELS:
        raise ValueError(f"{path}: unexpected label order {names}")
    if not (len(ids) == len(set(ids)) == len(metadata) == len(labels) == len(embeddings)):
        raise ValueError(f"{path}: IDs, metadata, labels and embeddings are misaligned")
    if embeddings.ndim != 2 or embeddings.shape[1] != 2048 or not torch.isfinite(embeddings).all():
        raise ValueError(f"{path}: expected finite [N,2048] Qwen3B embeddings")
    if any(not sample_id.startswith(f"{split}_") for sample_id in ids):
        raise ValueError(f"{path}: sample IDs do not all belong to {split}")
    config = payload.get("config", {})
    if config.get("model_id") != "Qwen/Qwen2.5-VL-3B-Instruct" or config.get("detector_package") != "py-feat 2.1.5":
        raise ValueError(f"{path}: unexpected extractor config")
    if config.get("detector_cudnn_enabled") is not False:
        raise ValueError(f"{path}: expected the corrected Detectorv2 run")
    statuses = [m.get("speaker_face", {}).get("status") for m in metadata]
    if any(status not in ("likely_speaker", "ambiguous_speaker", "no_face") for status in statuses):
        raise ValueError(f"{path}: unknown speaker-face status")
    keep = [i for i, status in enumerate(statuses) if status == "likely_speaker"]
    if not keep:
        raise ValueError(f"{path}: no likely-speaker samples")
    selected = torch.tensor(keep, dtype=torch.long)
    filtered = {
        "embeddings": embeddings.index_select(0, selected),
        "labels": labels.index_select(0, selected),
        "sample_ids": [ids[i] for i in keep],
        "metadata": [metadata[i] for i in keep],
        "label_names": names,
        "config": {**config, "sample_filter": "speaker_face.status == likely_speaker"},
        "errors": payload.get("errors", []),
    }
    target.parent.mkdir(parents=True, exist_ok=True)
    torch.save(filtered, target)
    return {"indexed_embeddings": len(ids), "likely_speaker": len(keep),
            "ambiguous_speaker": statuses.count("ambiguous_speaker"),
            "no_face": statuses.count("no_face"), "extractor_errors": len(payload.get("errors", []))}


def metrics(frame: pd.DataFrame, true_col: str, pred_col: str) -> dict:
    truth = frame[true_col].astype(str).to_numpy()
    pred = frame[pred_col].astype(str).to_numpy()
    return {
        "n": len(frame),
        "accuracy": float(accuracy_score(truth, pred)),
        "macro_f1": float(f1_score(truth, pred, labels=DEFAULT_LABELS, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(truth, pred, labels=DEFAULT_LABELS, average="weighted", zero_division=0)),
        "classification_report": classification_report(truth, pred, labels=DEFAULT_LABELS,
                                                       output_dict=True, zero_division=0),
        "confusion_matrix": confusion_matrix(truth, pred, labels=DEFAULT_LABELS).tolist(),
    }


def compare(split: str, mlp_dir: Path, baseline_dir: Path, output_dir: Path, likely_count: int) -> dict:
    new = pd.read_csv(mlp_dir / f"{split}_predictions.csv")
    old = pd.read_csv(baseline_dir / f"{split}_predictions.csv")
    required = {"sample_id", "true_label", "pred_label"}
    if not required.issubset(new) or not required.issubset(old):
        raise ValueError("Prediction CSVs lack required columns")
    if not new.sample_id.is_unique or not old.sample_id.is_unique:
        raise ValueError("Prediction CSVs contain duplicate sample IDs")
    if len(new) != likely_count:
        raise ValueError(f"{split}: MLP predictions do not cover every likely-speaker sample")
    paired = new.merge(old, on="sample_id", how="inner", suffixes=("_face_prompt", "_full_baseline"),
                       validate="one_to_one")
    if paired.empty:
        raise ValueError(f"{split}: no paired predictions")
    if not paired.true_label_face_prompt.equals(paired.true_label_full_baseline):
        raise ValueError(f"{split}: true labels disagree between experiments")
    paired = paired[["sample_id", "true_label_face_prompt", "pred_label_face_prompt",
                     "pred_label_full_baseline"]].rename(columns={"true_label_face_prompt": "true_label"})
    paired["face_prompt_correct"] = paired.pred_label_face_prompt == paired.true_label
    paired["full_baseline_correct"] = paired.pred_label_full_baseline == paired.true_label
    paired.to_csv(output_dir / f"aligned_{split}_predictions.csv", index=False)
    new_metrics = metrics(paired, "true_label", "pred_label_face_prompt")
    old_metrics = metrics(paired, "true_label", "pred_label_full_baseline")
    return {
        "likely_speaker_predictions": len(new),
        "paired_samples": len(paired),
        "likely_samples_missing_baseline": len(new) - len(paired),
        "face_prompt_retrained_mlp": new_metrics,
        "full_video_saved_baseline_mlp": old_metrics,
        "delta_face_minus_baseline": {
            key: new_metrics[key] - old_metrics[key]
            for key in ("accuracy", "macro_f1", "weighted_f1")
        },
        "paired_correctness": {
            "both_correct": int((paired.face_prompt_correct & paired.full_baseline_correct).sum()),
            "face_only_correct": int((paired.face_prompt_correct & ~paired.full_baseline_correct).sum()),
            "baseline_only_correct": int((~paired.face_prompt_correct & paired.full_baseline_correct).sum()),
            "both_wrong": int((~paired.face_prompt_correct & ~paired.full_baseline_correct).sum()),
        },
    }


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    filtered_dir = args.output_dir / "filtered_embeddings"
    coverage = {}
    for split in SPLITS:
        source = locate_embedding(args.results_dir, split, args.output_dir / "staged_inputs")
        target = filtered_dir / f"meld_{split}_likely.pt"
        coverage[split] = filter_likely(source, split, target)
        print(f"{split}: {coverage[split]}", flush=True)

    mlp_dir = args.output_dir / "likely_speaker_mlp"
    command = [sys.executable, str(REPO / "scripts/classification/train_mlp.py"),
               "--train-pt", str(filtered_dir / "meld_train_likely.pt"),
               "--dev-pt", str(filtered_dir / "meld_dev_likely.pt"),
               "--test-pt", str(filtered_dir / "meld_test_likely.pt"),
               "--output-dir", str(mlp_dir),
               "--epochs", str(args.epochs), "--seed", str(args.seed), "--device", args.device]
    subprocess.run(command, check=True)

    comparison = {"filter": "speaker_face.status == likely_speaker", "coverage": coverage,
                  "label_order": DEFAULT_LABELS, "mlp_epochs": args.epochs, "mlp_seed": args.seed}
    for split in ("dev", "test"):
        comparison[split] = compare(split, mlp_dir, args.baseline_dir, args.output_dir,
                                    coverage[split]["likely_speaker"])
        delta = comparison[split]["delta_face_minus_baseline"]
        print(f"{split}: paired n={comparison[split]['paired_samples']}, "
              f"face - full baseline accuracy={delta['accuracy']:+.4f}, "
              f"macro F1={delta['macro_f1']:+.4f}", flush=True)
    (args.output_dir / "aligned_comparison.json").write_text(json.dumps(comparison, indent=2), encoding="utf-8")
    print(f"Saved results to {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
