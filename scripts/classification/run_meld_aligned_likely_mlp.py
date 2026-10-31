#!/usr/bin/env python3
"""Retrain face-prompt and full-video MELD MLPs on identical likely-speaker IDs."""

from __future__ import annotations

import argparse
import json
import pickle
import subprocess
import sys
from pathlib import Path, PurePosixPath

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pandas as pd
import torch

from scripts.classification.run_meld_detectorv2_likely_mlp import locate_embedding, metrics
from scripts.classification.train_mlp import DEFAULT_LABELS

REPO = Path(__file__).resolve().parents[2]
SPLITS = ("train", "dev", "test")


class _CrossPlatformUnpickler(pickle.Unpickler):
    def find_class(self, module: str, name: str):
        if module == "pathlib" and name == "PosixPath":
            return PurePosixPath
        return super().find_class(module, name)


class _CrossPlatformPickle:
    Unpickler = _CrossPlatformUnpickler


def load_embedding(path: Path) -> dict:
    # Original baseline payloads contain PosixPath metadata, which Windows
    # cannot instantiate using the default pickle loader.
    return torch.load(path, map_location="cpu", weights_only=False,
                      pickle_module=_CrossPlatformPickle)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--face-results-dir", type=Path, required=True)
    parser.add_argument("--baseline-embeddings-dir", type=Path,
                        default=REPO / "experiments/baselines/qwen2_5_vl_3b_shared/embeddings")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default="auto")
    return parser.parse_args()


def check_payload(payload: dict, path: Path) -> dict[str, int]:
    ids = [str(value) for value in payload["sample_ids"]]
    if payload.get("label_names") != DEFAULT_LABELS:
        raise ValueError(f"{path}: unexpected label order")
    if len(ids) != len(set(ids)) or any(
        len(payload[key]) != len(ids) for key in ("embeddings", "labels", "metadata")
    ):
        raise ValueError(f"{path}: sample IDs or fields are misaligned")
    if payload["embeddings"].ndim != 2 or payload["embeddings"].shape[1] != 2048:
        raise ValueError(f"{path}: expected [N,2048] embeddings")
    if not torch.isfinite(payload["embeddings"]).all():
        raise ValueError(f"{path}: nonfinite embeddings")
    return {sample_id: i for i, sample_id in enumerate(ids)}


def save_subset(payload: dict, indices: list[int], target: Path, source: str) -> None:
    ix = torch.tensor(indices, dtype=torch.long)
    subset = {
        "embeddings": payload["embeddings"].index_select(0, ix),
        "labels": payload["labels"].index_select(0, ix),
        "sample_ids": [payload["sample_ids"][i] for i in indices],
        "metadata": [payload["metadata"][i] for i in indices],
        "label_names": payload["label_names"],
        "config": {**payload.get("config", {}), "alignment": "DetectorV2 likely_speaker and clean full-video ID intersection",
                   "source": source},
    }
    target.parent.mkdir(parents=True, exist_ok=True)
    torch.save(subset, target)


def train(model_name: str, aligned_dir: Path, output_dir: Path, args: argparse.Namespace) -> None:
    command = [sys.executable, str(REPO / "scripts/classification/train_mlp.py")]
    for split in SPLITS:
        command.extend((f"--{split}-pt", str(aligned_dir / model_name / f"meld_{split}.pt")))
    command.extend(("--output-dir", str(output_dir / model_name), "--epochs", str(args.epochs),
                    "--seed", str(args.seed), "--device", args.device))
    print(f"Training {model_name}", flush=True)
    subprocess.run(command, check=True)


def compare(split: str, output_dir: Path, expected_ids: list[str]) -> dict:
    face = pd.read_csv(output_dir / "face_prompt" / f"{split}_predictions.csv")
    full = pd.read_csv(output_dir / "full_video" / f"{split}_predictions.csv")
    if not face.sample_id.is_unique or not full.sample_id.is_unique:
        raise ValueError(f"{split}: duplicate prediction IDs")
    if set(face.sample_id) != set(expected_ids) or set(full.sample_id) != set(expected_ids):
        raise ValueError(f"{split}: predictions do not cover exactly the aligned IDs")
    paired = face[["sample_id", "true_label", "pred_label"]].merge(
        full[["sample_id", "true_label", "pred_label"]], on="sample_id", how="inner",
        suffixes=("_face", "_full"), validate="one_to_one")
    if not paired.true_label_face.equals(paired.true_label_full):
        raise ValueError(f"{split}: truth labels disagree")
    paired = paired.rename(columns={"true_label_face": "true_label",
                                    "pred_label_face": "face_prompt_prediction",
                                    "pred_label_full": "full_video_prediction"}).drop(columns="true_label_full")
    paired["face_correct"] = paired.true_label == paired.face_prompt_prediction
    paired["full_correct"] = paired.true_label == paired.full_video_prediction
    paired.to_csv(output_dir / f"aligned_{split}_predictions.csv", index=False)
    face_metrics = metrics(paired, "true_label", "face_prompt_prediction")
    full_metrics = metrics(paired, "true_label", "full_video_prediction")
    return {
        "n": len(paired),
        "face_prompt_mlp": face_metrics,
        "full_video_mlp": full_metrics,
        "delta_face_minus_full": {key: face_metrics[key] - full_metrics[key]
                                  for key in ("accuracy", "macro_f1", "weighted_f1")},
        "paired_correctness": {
            "both_correct": int((paired.face_correct & paired.full_correct).sum()),
            "face_only_correct": int((paired.face_correct & ~paired.full_correct).sum()),
            "full_only_correct": int((~paired.face_correct & paired.full_correct).sum()),
            "both_wrong": int((~paired.face_correct & ~paired.full_correct).sum()),
        },
    }


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    aligned_dir = args.output_dir / "aligned_embeddings"
    coverage = {}
    aligned_ids = {}
    for split in SPLITS:
        face_path = locate_embedding(args.face_results_dir, split, args.output_dir / "staged_face_inputs")
        full_path = args.baseline_embeddings_dir / f"meld_{split}_clean.pt"
        face = load_embedding(face_path)
        full = load_embedding(full_path)
        face_index = check_payload(face, face_path)
        full_index = check_payload(full, full_path)
        if face.get("config", {}).get("detector_cudnn_enabled") is not False:
            raise ValueError(f"{face_path}: expected the corrected DetectorV2 run")
        likely_ids = [sample_id for sample_id in face["sample_ids"]
                      if face["metadata"][face_index[str(sample_id)]].get("speaker_face", {}).get("status")
                      == "likely_speaker"]
        shared_ids = [str(sample_id) for sample_id in likely_ids if str(sample_id) in full_index]
        if not shared_ids:
            raise ValueError(f"{split}: no aligned likely-speaker IDs")
        face_ix = [face_index[sample_id] for sample_id in shared_ids]
        full_ix = [full_index[sample_id] for sample_id in shared_ids]
        if not torch.equal(face["labels"][face_ix].long(), full["labels"][full_ix].long()):
            raise ValueError(f"{split}: aligned labels disagree")
        save_subset(face, face_ix, aligned_dir / "face_prompt" / f"meld_{split}.pt", str(face_path))
        save_subset(full, full_ix, aligned_dir / "full_video" / f"meld_{split}.pt", str(full_path))
        aligned_ids[split] = shared_ids
        coverage[split] = {"face_embeddings": len(face_index), "clean_full_video_embeddings": len(full_index),
                           "likely_speaker": len(likely_ids), "aligned_likely_speaker": len(shared_ids),
                           "likely_speaker_missing_clean_full_video": len(likely_ids) - len(shared_ids)}
        print(f"{split}: {coverage[split]}", flush=True)

    for model_name in ("face_prompt", "full_video"):
        train(model_name, aligned_dir, args.output_dir, args)

    report = {"filter": "DetectorV2 speaker_face.status == likely_speaker",
              "alignment": "same clean full-video and face-prompt IDs in each train/dev/test split",
              "coverage": coverage, "epochs": args.epochs, "seed": args.seed, "label_order": DEFAULT_LABELS}
    for split in ("dev", "test"):
        report[split] = compare(split, args.output_dir, aligned_ids[split])
        print(f"{split}: n={report[split]['n']} delta face - full "
              f"accuracy={report[split]['delta_face_minus_full']['accuracy']:+.4f} "
              f"macro F1={report[split]['delta_face_minus_full']['macro_f1']:+.4f}", flush=True)
    (args.output_dir / "comparison.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Saved results to {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
