#!/usr/bin/env python3
"""Render two full-baseline attention examples and collect compact 7B results.

Run from codes/Emo-benchmark on Rorqual, with ``--root ../..``. The output is an
ordinary directory ready for the user to zip or download.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


REPO = Path(__file__).resolve().parents[2]
RESULT_DIRS = (
    "qwen2_5_vl_7b_mosei_multilabel_emotion_prompt_cache_success",
    "qwen2_5_vl_7b_mosei_divprune_curve_multilabel_emotion_prompt_cache_success",
    "qwen2_5_vl_7b_mosei_random_curve_seed96_multilabel_emotion_prompt_cache_success",
    "qwen2_5_vl_7b_mosei_sparsevlm_curve_multilabel_emotion_prompt_cache_success",
    "qwen2_5_vl_7b_mosei_vispruner_curve_multilabel_emotion_prompt_cache_success",
    "qwen2_5_vl_7b_mosei_visionzip_curve_multilabel_emotion_prompt_cache_success",
    "qwen2_5_vl_7b_mosei_lm_head_four_conditions_test",
    "qwen2_5_vl_7b_mosei_retained_visual_utterance_lm_head_test",
    "qwen2_5_vl_7b_meld_emotion_prompt_cache_success",
    "qwen2_5_vl_7b_meld_divprune_curve_emotion_prompt_cache_success",
    "qwen2_5_vl_7b_meld_random_curve_seed64_emotion_prompt_cache_success",
    "qwen2_5_vl_7b_meld_sparsevlm_curve_emotion_prompt_cache_success",
    "qwen2_5_vl_7b_meld_vispruner_curve_emotion_prompt_cache_success",
    "qwen2_5_vl_7b_meld_visionzip_curve_emotion_prompt_cache_success",
    "qwen2_5_vl_7b_meld_lm_head_four_conditions_test",
    "qwen2_5_vl_7b_meld_retained_visual_utterance_lm_head_test",
)
KEEP_SUFFIXES = {".csv", ".json", ".png"}
SKIP_PARTS = {
    "embeddings", "chunks", "checkpoints", "indexes", "attention_heatmaps",
    "__pycache__",
}
BASELINE_DIRS = {
    "mosei": RESULT_DIRS[0],
    "meld": RESULT_DIRS[8],
}


def choose_sample_id(root: Path, dataset: str) -> str:
    """Choose the first baseline test clip also listed as a successful decode."""
    index = root / "results" / BASELINE_DIRS[dataset] / "indexes" / f"{dataset}_test_index.csv"
    manifest = root / "datasets" / dataset / "decoded_qwen_6fps_224_v1" / "successful_decodes.csv"
    with manifest.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if not {"sample_id", "split"} <= set(reader.fieldnames or ()):
            raise ValueError(f"{manifest}: expected sample_id and split columns")
        successful = {row["sample_id"] for row in reader if row["split"] == "test"}
    with index.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if not {"sample_id", "split"} <= set(reader.fieldnames or ()):
            raise ValueError(f"{index}: expected sample_id and split columns")
        for row in reader:
            if row["split"] == "test" and row["sample_id"] in successful:
                return row["sample_id"]
    raise ValueError(f"No test clip in both {index} and {manifest}")


def result_files(results_root: Path) -> list[tuple[Path, Path]]:
    """Return (source, relative destination), with exact experiment selection."""
    missing = [name for name in RESULT_DIRS if not (results_root / name).is_dir()]
    if missing:
        raise FileNotFoundError("Missing expected result directories:\n  " + "\n  ".join(missing))
    selected = []
    for name in RESULT_DIRS:
        source_dir = results_root / name
        for source in sorted(source_dir.rglob("*")):
            relative = source.relative_to(source_dir)
            if (not source.is_file() or source.is_symlink()
                    or source.suffix.lower() not in KEEP_SUFFIXES
                    or any(part in SKIP_PARTS for part in relative.parts[:-1])):
                continue
            selected.append((source, Path("results") / name / relative))
    return selected


def visualize(root: Path, dataset: str, sample_id: str, destination: Path,
              verify_only: bool = False) -> None:
    script = REPO / "cluster" / f"visualize_{dataset}_full_attention.py"
    command = [sys.executable, "-u", str(script), "--root", str(root),
               "--split", "test", f"--sample-id={sample_id}",
               "--output-dir", str(destination)]
    if dataset == "mosei":
        command.extend(("--prompt-style", "emotion_multilabel_task"))
    if verify_only:
        command.append("--verify-only")
    print("Running:", " ".join(command), flush=True)
    subprocess.run(command, cwd=REPO, check=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True,
                        help="Project root containing results/, datasets/, and models/")
    parser.add_argument("--output-dir", type=Path,
                        help="New directory for the download bundle; default ROOT/cvpr_results_bundle")
    parser.add_argument("--mosei-sample-id", help="Override automatic MOSEI test clip selection")
    parser.add_argument("--meld-sample-id", help="Override automatic MELD test clip selection")
    parser.add_argument("--skip-attention", action="store_true",
                        help="Collect existing result files without GPU visualization")
    parser.add_argument("--list-only", action="store_true",
                        help="Show selected file counts and validate samples without copying or GPU work")
    args = parser.parse_args()
    root = args.root.expanduser().resolve()
    results_root = root / "results"
    output_dir = (args.output_dir or root / "cvpr_results_bundle").expanduser().resolve()
    if output_dir == results_root or results_root in output_dir.parents:
        parser.error("Output directory must be outside ROOT/results")
    if output_dir.exists() and not args.list_only:
        parser.error(f"Output directory already exists: {output_dir}; choose a new path")
    files = result_files(results_root)
    counts = {name: 0 for name in RESULT_DIRS}
    for _, relative in files:
        counts[relative.parts[1]] += 1
    for name, count in counts.items():
        print(f"{name}: {count} CSV/JSON/PNG files", flush=True)
    total_bytes = sum(source.stat().st_size for source, _ in files)
    print(f"Selected {len(files)} files ({total_bytes / (1024 ** 2):.1f} MiB)", flush=True)
    if not args.skip_attention:
        mosei_sample_id = args.mosei_sample_id or choose_sample_id(root, "mosei")
        meld_sample_id = args.meld_sample_id or choose_sample_id(root, "meld")
        print(f"Attention samples: MOSEI {mosei_sample_id}; MELD {meld_sample_id}", flush=True)
        # Catch sample/index/cache errors before loading a GPU model.
        for dataset, sample_id in (("mosei", mosei_sample_id),
                                   ("meld", meld_sample_id)):
            visualize(root, dataset, sample_id, output_dir / "attention" / dataset,
                      verify_only=True)
    if args.list_only:
        print(f"Would collect {len(files)} files into {output_dir}", flush=True)
        return 0

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f".{output_dir.name}.", dir=output_dir.parent) as temp:
        staging = Path(temp) / "bundle"
        staging.mkdir()
        for source, relative in files:
            target = staging / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
        if not args.skip_attention:
            for dataset, sample_id in (("mosei", mosei_sample_id),
                                       ("meld", meld_sample_id)):
                attention_dir = staging / "attention" / dataset
                visualize(root, dataset, sample_id, attention_dir)
                # Raw arrays can be regenerated from the cached clip and are large.
                (attention_dir / "attention_arrays.npz").unlink(missing_ok=True)
        manifest = {
            "source_root": str(root),
            "datasets": ["mosei_multilabel", "meld"],
            "result_directories": list(RESULT_DIRS),
            "file_counts_by_result": counts,
            "result_file_count": len(files),
            "result_total_bytes": total_bytes,
            "attention_samples": (None if args.skip_attention else {
                "mosei": mosei_sample_id, "meld": meld_sample_id}),
            "included_suffixes": sorted(KEEP_SUFFIXES),
            "excluded_directories": sorted(SKIP_PARTS),
        }
        (staging / "bundle_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        os.replace(staging, output_dir)
    print(f"Bundle ready: {output_dir} ({len(files)} result files plus attention overlays)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
