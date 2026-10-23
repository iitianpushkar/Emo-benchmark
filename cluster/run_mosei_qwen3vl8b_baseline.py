#!/usr/bin/env python3
"""Run the full Qwen3-VL-8B frozen-embedding baseline on CMU-MOSEI.

Submit this command inside a single-H100 allocation. It never downloads data or
weights. Completed extraction chunks are reused when the job is restarted.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shlex
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from cluster.preflight import verify_model, verify_mosei  # noqa: E402

SPLIT_COUNTS = {"train": 16326, "valid": 1871, "test": 4659}
MODEL_ID = "Qwen/Qwen3-VL-8B-Instruct"
HIDDEN_SIZE = 4096


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_revision() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def run(command: list[str], env: dict[str, str]) -> None:
    print("+", shlex.join(command), flush=True)
    subprocess.run(command, cwd=REPO_ROOT, env=env, check=True)


def index_ids(path: Path, expected: int) -> list[str]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    ids = [row["sample_id"] for row in rows]
    if len(ids) != expected or len(set(ids)) != expected:
        raise ValueError(f"{path}: expected {expected} unique samples, found {len(ids)}")
    if any(row["video_exists"].lower() != "true" for row in rows):
        raise ValueError(f"{path}: some indexed videos are missing")
    return ids


def verify_embeddings(path: Path, expected_ids: list[str]) -> None:
    import torch

    payload = torch.load(path, map_location="cpu", weights_only=False)
    ids = [str(value) for value in payload["sample_ids"]]
    embeddings = payload["embeddings"]
    errors = payload.get("errors", [])
    if errors:
        raise ValueError(f"{path}: {len(errors)} extraction errors; first examples: {errors[:3]}")
    if embeddings.ndim != 2 or embeddings.shape != (len(expected_ids), HIDDEN_SIZE):
        raise ValueError(f"{path}: expected embedding shape {(len(expected_ids), HIDDEN_SIZE)}, got {tuple(embeddings.shape)}")
    if len(set(ids)) != len(ids) or set(ids) != set(expected_ids):
        raise ValueError(f"{path}: sample IDs differ from the split index")
    if not torch.isfinite(embeddings).all():
        raise ValueError(f"{path}: embeddings contain NaN or infinity")
    print(f"Verified {path}: {len(ids)} complete embeddings", flush=True)


def metrics_complete(path: Path) -> bool:
    if not path.is_file() or not (path.parent / "best_mlp.pt").is_file() or not (path.parent / "test_predictions.csv").is_file():
        return False
    try:
        report = json.loads(path.read_text())
        coverage = report["coverage"]
        return (
            all(coverage[split]["embedded"] == expected and coverage[split]["missing"] == 0
                for split, expected in (("train", 16326), ("validation", 1871), ("test", 4659)))
            and report["test"]["sample_count"] == SPLIT_COUNTS["test"]
        )
    except (OSError, ValueError, KeyError, TypeError):
        return False


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=REPO_ROOT.parents[1],
                        help="Shared parent of codes/, datasets/, models/, and results/")
    parser.add_argument("--model-dir", type=Path, help="Local Qwen3-VL-8B-Instruct folder (default: ROOT/models/qwen)")
    parser.add_argument("--mosei-root", type=Path, help="Local MOSEI folder (default: ROOT/datasets/mosei)")
    parser.add_argument("--output-dir", type=Path, help="Run artifacts (default: ROOT/results/qwen3_vl_8b_mosei)")
    parser.add_argument("--chunk-size", type=int, default=500)
    parser.add_argument("--verify-only", action="store_true", help="Validate local inputs without starting GPU work")
    args = parser.parse_args()
    if args.chunk_size <= 0:
        parser.error("--chunk-size must be positive")

    root = args.root.expanduser().resolve()
    video_protocol = json.loads((Path(__file__).with_name("mosei_video_protocol.json")).read_text())
    model_dir = (args.model_dir or root / "models" / "qwen").expanduser().resolve()
    mosei_root = (args.mosei_root or root / "datasets" / "mosei").expanduser().resolve()
    output_dir = (args.output_dir or root / "results" / "qwen3_vl_8b_mosei").expanduser().resolve()
    model = verify_model(model_dir, "qwen")
    data = verify_mosei(mosei_root)
    mosei_root = Path(data["root"]).resolve()
    if args.verify_only:
        print(json.dumps({"model_id": MODEL_ID, "model": model, "data": data,
                          "video_protocol": video_protocol}, indent=2))
        return 0

    import torch

    if not torch.cuda.is_available() or not any(
        "H100" in torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())
    ):
        raise RuntimeError("This full baseline requires an allocated H100 CUDA GPU")

    config = {
        "model_id": MODEL_ID,
        "model_dir": str(model_dir),
        "model_config_sha256": sha256(model_dir / "config.json"),
        "model_index_sha256": sha256(model_dir / "model.safetensors.index.json"),
        "mosei_root": str(mosei_root),
        "aligned_csv_sha256": sha256(mosei_root / "csv" / "mosei_aligned.csv"),
        "code_revision": git_revision(),
        "chunk_size": args.chunk_size,
        "extraction": {**video_protocol, "modality_mode": "video_text",
                       "pooling": "last", "layer": -1, "save_dtype": "float32"},
        "target": "six independent binary emotions; positive iff score > 0",
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    config_path = output_dir / "run_config.json"
    if config_path.is_file():
        previous = json.loads(config_path.read_text())
        if previous != config:
            raise ValueError(f"Run configuration changed; use a new --output-dir or inspect {config_path}")
    else:
        config_path.write_text(json.dumps(config, indent=2) + "\n")

    env = os.environ.copy()
    env["HF_HUB_OFFLINE"] = "1"
    index_dir = output_dir / "indexes"
    run([sys.executable, str(REPO_ROOT / "scripts" / "data" / "prepare_mosei.py"),
         "--mosei-root", str(mosei_root),
         "--output-dir", str(index_dir)], env)

    for split, expected in SPLIT_COUNTS.items():
        index = index_dir / f"mosei_{split}_index.csv"
        ids = index_ids(index, expected)
        chunks_dir = output_dir / "chunks" / split
        embedding_pt = output_dir / "embeddings" / f"mosei_{split}.pt"
        for start in range(0, expected, args.chunk_size):
            end = min(start + args.chunk_size, expected)
            chunk = chunks_dir / f"{split}_{start:05d}_{end:05d}.pt"
            if chunk.is_file():
                import torch

                payload = torch.load(chunk, map_location="cpu", weights_only=False)
                expected_chunk_ids = ids[start:end]
                if (not payload.get("errors")
                        and len(set(payload.get("sample_ids", []))) == end - start
                        and set(payload["sample_ids"]) == set(expected_chunk_ids)
                        and tuple(payload["embeddings"].shape) == (end - start, HIDDEN_SIZE)
                        and payload.get("config", {}).get("model_dir") == str(model_dir)
                        and payload.get("config", {}).get("start") == start
                        and payload.get("config", {}).get("limit") == end - start
                        and payload.get("config", {}).get("video_protocol") == video_protocol
                        and payload.get("config", {}).get("retain_ratio") == 1.0
                        and payload.get("config", {}).get("pruning") == "none"):
                    print(f"Reusing complete chunk {chunk}", flush=True)
                    continue
            run([sys.executable, str(REPO_ROOT / "cluster" / "extract_mosei_qwen3vl.py"),
                 "--index-csv", str(index), "--output-pt", str(chunk),
                 "--model-dir", str(model_dir), "--start", str(start),
                 "--limit", str(end - start)], env)
        run([sys.executable, "-m", "scripts", "merge_embedding_chunks",
             "--chunks-dir", str(chunks_dir), "--output-pt", str(embedding_pt),
             "--pattern", f"{split}_*.pt"], env)
        verify_embeddings(embedding_pt, ids)

    metrics_path = output_dir / "outputs" / "metrics.json"
    if not metrics_complete(metrics_path):
        run([sys.executable, str(REPO_ROOT / "scripts" / "classification" / "mosei_mlp.py"), "train",
             "--train-pt", str(output_dir / "embeddings" / "mosei_train.pt"),
             "--train-index", str(index_dir / "mosei_train_index.csv"),
             "--valid-pt", str(output_dir / "embeddings" / "mosei_valid.pt"),
             "--valid-index", str(index_dir / "mosei_valid_index.csv"),
             "--test-pt", str(output_dir / "embeddings" / "mosei_test.pt"),
             "--test-index", str(index_dir / "mosei_test_index.csv"),
             "--output-dir", str(output_dir / "outputs"), "--device", "cuda"], env)
    if not metrics_complete(metrics_path):
        raise RuntimeError(f"Classifier metrics incomplete: {metrics_path}")
    (output_dir / "completed_utc.txt").write_text(datetime.now(timezone.utc).isoformat() + "\n")
    print(f"Full baseline complete: {metrics_path}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
