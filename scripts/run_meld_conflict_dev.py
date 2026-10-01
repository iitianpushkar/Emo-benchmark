#!/usr/bin/env python3
"""Prepare, extract, merge, and evaluate MELD dev conflict pairs."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
import sys
from collections import Counter
from pathlib import Path

REQUIRED_PAIR_COLUMNS = {
    "pair_id",
    "split",
    "pair_type",
    "is_conflict",
    "utterance_source_emotion",
    "video_source_emotion",
    "utterance_emotion_id",
    "video_emotion_id",
    "utterance_sample_id",
    "utterance_dialogue_id",
    "video_dialogue_id",
    "utterance_id",
    "utterance",
    "utterance_source_sentiment",
    "video_file",
    "video_path",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Qwen + fixed-MLP evaluation on MELD dev conflicts.")
    parser.add_argument(
        "--pair-csv",
        type=Path,
        default=Path("outputs/meld_conflict_dataset/meld_dev_conflict_matrix.csv"),
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("outputs/qwen2_5_vl_3b_shared/best_mlp.pt"),
        help="Joint video+utterance best_mlp.pt.",
    )
    parser.add_argument(
        "--work-dir",
        type=Path,
        default=Path("outputs/qwen2_5_vl_3b_conflict_dev"),
    )
    parser.add_argument(
        "--stage",
        choices=["all", "prepare", "extract", "evaluate"],
        default="all",
        help="Run the complete pipeline or one stage. The extract stage includes preparation and merging.",
    )
    parser.add_argument("--model-id", default="Qwen/Qwen2.5-VL-3B-Instruct")
    parser.add_argument("--chunk-size", type=int, default=250)
    parser.add_argument("--limit", type=int, default=None, help="Optional extraction smoke-test limit.")
    parser.add_argument("--fps", type=float, default=6.0)
    parser.add_argument("--min-frames", type=int, default=4)
    parser.add_argument("--max-frames", type=int, default=64)
    parser.add_argument("--frame-size", type=int, default=224)
    parser.add_argument("--video-max-pixels", type=int, default=None)
    parser.add_argument("--pooling", choices=["last", "mean", "max"], default="last")
    parser.add_argument("--prompt-style", choices=["emotion_task", "utterance_only"], default="emotion_task")
    parser.add_argument("--layer", type=int, default=-1)
    parser.add_argument("--save-dtype", choices=["float16", "float32"], default="float32")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"], default="auto")
    parser.add_argument("--skip-decord-precheck", action="store_true")
    parser.add_argument("--no-generation-prompt", action="store_true")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--require-complete", action="store_true")
    return parser.parse_args()


def script_path(name: str) -> Path:
    return Path(__file__).resolve().parent / name


def run(command: list[str]) -> None:
    print("\n$ " + " ".join(command), flush=True)
    subprocess.run(command, check=True)


def extraction_signature(args: argparse.Namespace, pair_csv: Path) -> str:
    pair_digest = hashlib.sha256(pair_csv.read_bytes()).hexdigest()
    configuration = {
        "pair_csv_sha256": pair_digest,
        "model_id": args.model_id,
        "chunk_size": args.chunk_size,
        "limit": args.limit,
        "fps": args.fps,
        "min_frames": args.min_frames,
        "max_frames": args.max_frames,
        "frame_size": args.frame_size,
        "video_max_pixels": args.video_max_pixels,
        "pooling": args.pooling,
        "prompt_style": args.prompt_style,
        "layer": args.layer,
        "save_dtype": args.save_dtype,
        "add_generation_prompt": not args.no_generation_prompt,
    }
    serialized = json.dumps(configuration, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(serialized).hexdigest()[:12]


def parse_bool(value: object) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes"}


def prepare_dev_conflict_index(pair_csv: Path, output_csv: Path) -> list[dict[str, object]]:
    with pair_csv.open(newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        fieldnames = list(reader.fieldnames or [])
        source_rows = list(reader)

    missing = sorted(REQUIRED_PAIR_COLUMNS.difference(fieldnames))
    if missing:
        raise ValueError(f"{pair_csv} is missing required columns: {missing}")

    rows = [
        row
        for row in source_rows
        if row["split"] == "dev" and row["pair_type"] == "cross_class_conflict"
    ]
    if not rows:
        raise ValueError(f"No dev cross_class_conflict rows found in {pair_csv}")
    pair_ids = [row["pair_id"] for row in rows]
    if len(pair_ids) != len(set(pair_ids)):
        raise ValueError("Conflict pair IDs must be unique")
    if not all(parse_bool(row["is_conflict"]) for row in rows):
        raise ValueError("Filtered conflict rows contain is_conflict=False")
    if any(row["utterance_source_emotion"] == row["video_source_emotion"] for row in rows):
        raise ValueError("Conflict rows contain matching source emotions")
    if any(row["utterance_dialogue_id"] == row["video_dialogue_id"] for row in rows):
        raise ValueError("Conflict donor videos must come from another dialogue")

    canonical_columns = [
        "sample_id",
        "split",
        "dialogue_id",
        "utterance_id",
        "utterance",
        "emotion",
        "emotion_id",
        "sentiment",
        "video_file",
        "video_path",
        "video_exists",
    ]
    extra_columns = [column for column in fieldnames if column not in canonical_columns]
    output_columns = canonical_columns + extra_columns
    prepared = []
    for row in rows:
        prepared_row: dict[str, object] = {
            "sample_id": row["pair_id"],
            "split": "dev_conflict",
            "dialogue_id": int(row["utterance_dialogue_id"]),
            "utterance_id": int(row["utterance_id"]),
            "utterance": row["utterance"],
            "emotion": row["utterance_source_emotion"],
            "emotion_id": int(row["utterance_emotion_id"]),
            "sentiment": row["utterance_source_sentiment"],
            "video_file": row["video_file"],
            "video_path": row["video_path"],
            "video_exists": True,
        }
        prepared_row.update({column: row[column] for column in extra_columns})
        prepared.append(prepared_row)

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=output_columns)
        writer.writeheader()
        writer.writerows(prepared)

    counts = Counter(
        (str(row["utterance_source_emotion"]), str(row["video_source_emotion"]))
        for row in prepared
    )
    if len(counts) != 42:
        raise ValueError(f"Expected all 42 ordered off-diagonal emotion pairs, found {len(counts)}")
    print(f"Prepared dev conflict index: {len(prepared):,} rows -> {output_csv}")
    print("Ordered emotion pairs: 42")
    return prepared


def main() -> None:
    args = parse_args()
    if args.chunk_size <= 0:
        raise ValueError("--chunk-size must be positive")
    if args.limit is not None and args.limit <= 0:
        raise ValueError("--limit must be positive")
    work_dir = args.work_dir.expanduser().resolve()
    work_dir.mkdir(parents=True, exist_ok=True)
    pair_csv = args.pair_csv.expanduser().resolve()
    signature = extraction_signature(args, pair_csv)
    run_dir = work_dir / f"run_{signature}"
    index_csv = run_dir / "meld_dev_conflict_index.csv"
    chunks_dir = run_dir / "chunks"
    embeddings_pt = run_dir / "meld_dev_conflict_embeddings.pt"
    evaluation_dir = run_dir / "evaluation"
    print(f"Run signature: {signature}")
    print(f"Run directory: {run_dir}")

    if args.stage in {"all", "prepare", "extract"}:
        prepared = prepare_dev_conflict_index(pair_csv, index_csv)
        if args.limit is not None:
            print(f"Extraction limit: {args.limit:,}/{len(prepared):,} prepared rows")
    if args.stage == "prepare":
        return

    if args.stage in {"all", "extract"}:
        extract_command = [
            sys.executable,
            str(script_path("run_qwen_shared_chunks.py")),
            "--index-csv",
            str(index_csv),
            "--chunks-dir",
            str(chunks_dir),
            "--chunk-prefix",
            "dev_conflict",
            "--chunk-size",
            str(args.chunk_size),
            "--skip-existing-complete",
            "--model-id",
            args.model_id,
            "--fps",
            str(args.fps),
            "--min-frames",
            str(args.min_frames),
            "--max-frames",
            str(args.max_frames),
            "--frame-size",
            str(args.frame_size),
            "--pooling",
            args.pooling,
            "--prompt-style",
            args.prompt_style,
            "--modality-mode",
            "video_text",
            "--save-dtype",
            args.save_dtype,
            "--layer",
            str(args.layer),
        ]
        if args.limit is not None:
            extract_command.extend(["--limit", str(args.limit)])
        if args.video_max_pixels is not None:
            extract_command.extend(["--video-max-pixels", str(args.video_max_pixels)])
        if args.skip_decord_precheck:
            extract_command.append("--skip-decord-precheck")
        if args.no_generation_prompt:
            extract_command.append("--no-generation-prompt")
        if args.trust_remote_code:
            extract_command.append("--trust-remote-code")
        if args.no_resume:
            extract_command.append("--no-resume")
        run(extract_command)

        run(
            [
                sys.executable,
                str(script_path("merge_embedding_chunks.py")),
                "--chunks-dir",
                str(chunks_dir),
                "--pattern",
                "dev_conflict_*.pt",
                "--output-pt",
                str(embeddings_pt),
            ]
        )
        if args.stage == "extract":
            return

    if not embeddings_pt.exists():
        raise FileNotFoundError(f"Conflict embeddings not found: {embeddings_pt}")
    evaluate_command = [
        sys.executable,
        str(script_path("evaluate_meld_conflicts.py")),
        "--checkpoint",
        str(args.checkpoint.expanduser().resolve()),
        "--embeddings-pt",
        str(embeddings_pt),
        "--pair-csv",
        str(pair_csv),
        "--output-dir",
        str(evaluation_dir),
        "--batch-size",
        str(args.batch_size),
        "--device",
        args.device,
    ]
    if args.require_complete:
        evaluate_command.append("--require-complete")
    run(evaluate_command)


if __name__ == "__main__":
    main()
