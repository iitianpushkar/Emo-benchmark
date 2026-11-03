#!/usr/bin/env python3
"""Train single-label MOSEI classifiers from existing baseline or curve embeddings."""

from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
CLASSIFIER = REPO_ROOT / "scripts" / "classification" / "mosei_single_label_mlp.py"
LABELS = ("happy", "sad", "anger", "surprise", "disgust", "fear", "neutral")


def run(command: list[str]) -> None:
    print("Running:", " ".join(command), flush=True)
    subprocess.run(command, cwd=REPO_ROOT, env={**os.environ, "PYTHONUNBUFFERED": "1"}, check=True)


def current(path: Path, *inputs: Path) -> bool:
    return (path.is_file() and all(path.stat().st_mtime >= item.stat().st_mtime
                                   for item in inputs))


def summarize_reports(root: Path, conditions: list[str], seeds: list[int],
                      source_kind: str, tie_policy: str, test_all: bool) -> dict:
    result = {"task": "single_max_seven_class", "class_order": list(LABELS),
              "tie_policy": tie_policy, "zero_policy": "neutral",
              "source_kind": source_kind, "test_evaluated": source_kind == "baseline" or test_all,
              "conditions": {}}
    for condition in conditions:
        folder = root / condition
        reports = [json.loads((folder / f"seed_{seed}" / "metrics.json").read_text())
                   for seed in seeds]
        metrics = {}
        for split in ("validation", "test"):
            if split == "test" and not result["test_evaluated"]:
                continue
            evaluations = ([report[split] for report in reports] if split == "validation" else
                           [json.loads((folder / "test_eval" / f"seed_{seed}" / "metrics.json").read_text())["metrics"]
                            for seed in seeds] if source_kind == "curve" else
                           [report[split] for report in reports])
            metrics[split] = {
                key: {"mean": statistics.mean(float(item[key]) for item in evaluations),
                      "sd": statistics.stdev(float(item[key]) for item in evaluations)
                      if len(evaluations) > 1 else 0.0}
                for key in ("accuracy", "macro_f1", "weighted_f1")
            }
        coverage = dict(reports[0]["coverage"])
        if source_kind == "curve" and result["test_evaluated"]:
            coverage["test"] = json.loads(
                (folder / "test_eval" / f"seed_{seeds[0]}" / "metrics.json").read_text())["coverage"]
        result["conditions"][condition] = {
            "coverage": coverage, "metrics": metrics,
            "seed_metrics": {str(seed): {"validation": report["validation"],
                                         **({"test": (json.loads((folder / "test_eval" / f"seed_{seed}" / "metrics.json").read_text())["metrics"]
                                                      if source_kind == "curve" else report["test"])}
                                            if result["test_evaluated"] else {})}
                             for seed, report in zip(seeds, reports)},
        }
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, required=True,
                        help="Existing baseline or pruning-curve results directory")
    parser.add_argument("--output-dir", type=Path,
                        help="Default: SOURCE_DIR/single_label")
    parser.add_argument("--conditions", nargs="+", help="Optional curve ratio folders, e.g. r100 r030")
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    parser.add_argument("--tie-policy", choices=("first", "exclude"), default="exclude")
    parser.add_argument("--test-all", action="store_true",
                        help="Evaluate curve test splits after the single-label validation run")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="cuda")
    parser.add_argument("--force", action="store_true", help="Retrain even if outputs are current")
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    if not args.seeds or len(set(args.seeds)) != len(args.seeds):
        parser.error("At least one unique seed is required")
    source = args.source_dir.expanduser().resolve()
    output = (args.output_dir or source / "single_label").expanduser().resolve()
    index_dir = source / "indexes"
    if not index_dir.is_dir():
        raise FileNotFoundError(index_dir)
    baseline = (source / "embeddings" / "mosei_train.pt").is_file()
    if baseline and args.conditions:
        parser.error("--conditions applies to curve results, not a baseline")
    if baseline and args.test_all:
        parser.error("Baseline test evaluation is automatic; omit --test-all")
    source_kind = "baseline" if baseline else "curve"
    conditions = (["baseline"] if baseline else
                  args.conditions or sorted(p.name for p in source.glob("r[0-9][0-9][0-9]") if p.is_dir()))
    if not conditions:
        raise ValueError(f"No baseline embeddings or curve ratio folders found in {source}")
    if len(conditions) != len(set(conditions)):
        parser.error("--conditions must be unique")
    if source_kind == "curve" and args.test_all:
        summary_path = output / "summary.json"
        if not summary_path.is_file():
            raise ValueError("Run the single-label validation conditions before --test-all")
        earlier = json.loads(summary_path.read_text())
        if (earlier.get("tie_policy") != args.tie_policy
                or not set(conditions) <= set(earlier.get("conditions", {}))):
            raise ValueError("The earlier single-label validation summary does not match these conditions and tie policy")
    splits = ("train", "valid", "test") if source_kind == "baseline" or args.test_all else ("train", "valid")
    inputs = {}
    for condition in conditions:
        folder = source if baseline else source / condition
        inputs[condition] = {}
        for split in splits:
            embedding = folder / "embeddings" / f"mosei_{split}.pt"
            index = index_dir / f"mosei_{split}_index.csv"
            if not embedding.is_file() or not index.is_file():
                raise FileNotFoundError(f"Missing embeddings or index: {embedding}, {index}")
            inputs[condition][split] = (embedding, index)
    print(json.dumps({"source": str(source), "source_kind": source_kind,
                      "output": str(output), "conditions": conditions, "splits": splits,
                      "seeds": args.seeds, "tie_policy": args.tie_policy,
                      "class_order": LABELS}, indent=2), flush=True)
    if args.verify_only:
        return 0
    import torch
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but no GPU is available; use --device cpu if appropriate")
    output.mkdir(parents=True, exist_ok=True)
    for condition in conditions:
        folder = output / condition
        train_pt, train_index = inputs[condition]["train"]
        valid_pt, valid_index = inputs[condition]["valid"]
        for seed in args.seeds:
            mlp_dir = folder / f"seed_{seed}"
            metrics_path = mlp_dir / "metrics.json"
            checkpoint = mlp_dir / "best_mlp.pt"
            training_current = (not args.force and checkpoint.is_file()
                                and current(metrics_path, train_pt, train_index,
                                            valid_pt, valid_index, CLASSIFIER,
                                            *((inputs[condition]["test"]) if baseline else ()))
                                and current(checkpoint, train_pt, train_index,
                                            valid_pt, valid_index, CLASSIFIER)
                                and (mlp_dir / "validation_predictions.csv").is_file()
                                and (not baseline or (mlp_dir / "test_predictions.csv").is_file())
                                and json.loads(metrics_path.read_text()).get("tie_policy") == args.tie_policy)
            if not training_current:
                command = [sys.executable, "-u", str(CLASSIFIER), "train",
                           "--train-pt", str(train_pt), "--train-index", str(train_index),
                           "--valid-pt", str(valid_pt), "--valid-index", str(valid_index),
                           "--output-dir", str(mlp_dir), "--seed", str(seed),
                           "--tie-policy", args.tie_policy, "--device", args.device]
                if baseline:
                    test_pt, test_index = inputs[condition]["test"]
                    command += ["--test-pt", str(test_pt), "--test-index", str(test_index)]
                run(command)
            else:
                print(f"Reusing single-label MLP seed {seed}: {mlp_dir}", flush=True)
            if source_kind == "curve" and args.test_all:
                test_pt, test_index = inputs[condition]["test"]
                test_dir = folder / "test_eval" / f"seed_{seed}"
                test_metrics = test_dir / "metrics.json"
                if (args.force or not current(test_metrics, checkpoint, test_pt, test_index, CLASSIFIER)
                        or not (test_dir / "predictions.csv").is_file()):
                    run([sys.executable, "-u", str(CLASSIFIER), "evaluate",
                         "--checkpoint", str(checkpoint), "--embeddings-pt", str(test_pt),
                         "--index", str(test_index), "--output-dir", str(test_dir),
                         "--device", args.device])
    report = summarize_reports(output, conditions, args.seeds, source_kind, args.tie_policy, args.test_all)
    (output / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    print(f"Single-label metrics: {output / 'summary.json'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
