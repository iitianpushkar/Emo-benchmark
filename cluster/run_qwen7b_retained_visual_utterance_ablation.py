#!/usr/bin/env python3
"""Paired test-set LM-head passes: retained video with and without utterance.

Ratios were selected from previously evaluated test curves and are exploratory.
MOSEI uses six independent Y/N queries; MELD uses one A-G query per clip.
"""

from __future__ import annotations

import argparse
from contextlib import nullcontext
import hashlib
import json
import os
from pathlib import Path
import sys

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from cluster.qwen2_5_divprune import DivPruneHook
from cluster.qwen_sparsevlm import SparseVLMHook
from cluster.qwen_vispruner_attention import QwenVisionAttentionCapture, QwenVisionZipCapture
from cluster.run_qwen7b_lm_head_visual_ablation import (
    atomic_csv, atomic_json, digest_rows, letter_map, messages_for,
    single_token_ids, score_candidates,
)
from cluster.meld_cache_subset import selected_cached_rows
from cluster.mosei_cache_subset import cached_multilabel_rows, resolve_cache_frames
from cluster.mosei_decoded_cache import identity, load_cached
from cluster.preflight import verify_meld, verify_model, verify_mosei
from cluster.verify_qwen_environment import require_cuda_gpu, verify_qwen_environment
from scripts.pruning.random_prune import sample_seed

RATIOS = {
    "mosei": {"divprune": .20, "random": .30, "sparsevlm": .15,
              "vispruner": .20, "visionzip": .15},
    "meld": {"divprune": .05, "random": .10, "sparsevlm": .15,
             "vispruner": .15, "visionzip": .10},
}
FIELDS = ("sample_id", "dataset", "method", "retention_ratio", "emotion_code", "emotion",
          "ground_truth", "video_only_probability", "video_utterance_probability",
          "probability_delta_with_utterance", "video_only_ground_truth_probability",
          "video_utterance_ground_truth_probability", "ground_truth_probability_delta",
          "video_only_prediction", "video_utterance_prediction", "full_visual_tokens",
          "retained_visual_tokens")
VERSION = "retained_visual_utterance_lm_head_v1"


def paired_prompt(dataset: str, utterance: str, code: str | None, include_utterance: bool) -> str:
    """Use identical instructions in both passes; change only the utterance line."""
    line = f"\nUtterance: {utterance}" if include_utterance else ""
    if dataset == "meld":
        choices = ", ".join(f"{letter}={name}" for letter, name in letter_map(dataset).items())
        return (f"Classify the speaker's emotion from the available evidence. "
                f"Choose exactly one code: {choices}.{line}\n"
                "Answer with exactly one capital letter, A through G.")
    assert code is not None
    return (f"Assess whether emotion {code}={letter_map(dataset)[code]} is present "
            f"in the speaker from the available evidence.{line}\n"
            "Answer exactly Y if present, or N if absent.")


def prepare_inputs(processor, model, video_path, protocol, prompt, images, videos, video_kwargs):
    text = processor.apply_chat_template(messages_for(video_path, protocol, prompt),
                                         tokenize=False, add_generation_prompt=True)
    return processor(text=[text], images=images, videos=[item[0] for item in videos],
                     video_metadata=[item[1] for item in videos], return_tensors="pt",
                     do_resize=False, **video_kwargs).to(model.device)


def pruned_probability(model, inputs, candidate_ids, torch, method, ratio, seed,
                       frozen_sparse_plan=None):
    attention = (QwenVisionAttentionCapture(model, inputs["video_grid_thw"])
                 if method == "vispruner" else
                 QwenVisionZipCapture(model, inputs["video_grid_thw"])
                 if method == "visionzip" else nullcontext())
    with torch.inference_mode(), attention as vision_attention:
        pruning = (SparseVLMHook(model, inputs["input_ids"], ratio,
                                 frozen_plan=frozen_sparse_plan)
                   if method == "sparsevlm" else
                   DivPruneHook(model, inputs["input_ids"], ratio, method=method,
                                random_seed=seed, vision_attention=vision_attention))
        with pruning:
            probabilities = score_candidates(model, inputs, candidate_ids, torch)
    selected = (pruning.stage_plan if method == "sparsevlm" and frozen_sparse_plan is None else
                pruning.selected_video_positions if method != "sparsevlm" else
                frozen_sparse_plan)
    return probabilities, selected, pruning.full_tokens, pruning.retained_tokens


def score_clip(row, dataset, method, ratio, model, processor, cache_dir, protocol, token_ids, torch,
               random_seed):
    utterance = " ".join(row["text"].split()) if dataset == "mosei" else row["utterance"]
    video_path = Path(row["video_path"] if dataset == "meld" else row["_video_path"])
    cached = load_cached(cache_dir, identity(row["sample_id"], video_path, protocol))
    if cached is None:
        raise RuntimeError(f"Cached frames disappeared for {row['sample_id']}")
    images, videos, video_kwargs = cached
    labels = letter_map(dataset)
    codes = labels if dataset == "mosei" else (None,)
    seed = sample_seed(random_seed, row["sample_id"]) if method == "random" else None
    output = []
    for code in codes:
        only_inputs = prepare_inputs(processor, model, video_path, protocol,
            paired_prompt(dataset, utterance, code, False), images, videos, video_kwargs)
        both_inputs = prepare_inputs(processor, model, video_path, protocol,
            paired_prompt(dataset, utterance, code, True), images, videos, video_kwargs)
        video_id = model.config.video_token_id
        only_video_positions = torch.where(only_inputs["input_ids"][0].eq(video_id))[0]
        both_video_positions = torch.where(both_inputs["input_ids"][0].eq(video_id))[0]
        if not torch.equal(only_video_positions, both_video_positions):
            raise ValueError("Paired prompts changed video-token positions")
        if not only_video_positions.numel():
            raise ValueError("Cached video produced no visual tokens")
        only, only_plan, full, retained = pruned_probability(
            model, only_inputs, token_ids, torch, method, ratio, seed)
        both, both_plan, full_both, retained_both = pruned_probability(
            model, both_inputs, token_ids, torch, method, ratio, seed,
            frozen_sparse_plan=only_plan if method == "sparsevlm" else None)
        if full != full_both or retained != retained_both:
            raise ValueError("Paired passes have different visual-token budgets")
        if method != "sparsevlm" and only_plan != both_plan:
            raise ValueError("Paired passes retained different visual-token positions")
        if dataset == "meld":
            truth = row["emotion"].lower()
            if truth not in labels.values():
                raise ValueError(f"Unknown MELD label: {truth}")
            only_prediction = list(labels.values())[max(range(7), key=only.__getitem__)]
            both_prediction = list(labels.values())[max(range(7), key=both.__getitem__)]
            items = [(letter, emotion, int(emotion == truth), only[i], both[i],
                      only_prediction, both_prediction) for i, (letter, emotion) in enumerate(labels.items())]
        else:
            emotion = labels[code]
            truth = int(float(row[emotion]) > 0)
            items = [(code, emotion, truth, only[0], both[0],
                      int(only[0] >= .5), int(both[0] >= .5))]
        for letter, emotion, truth, p_only, p_both, pred_only, pred_both in items:
            gt_only = (p_only if truth else 1 - p_only) if dataset == "mosei" else (p_only if truth else "")
            gt_both = (p_both if truth else 1 - p_both) if dataset == "mosei" else (p_both if truth else "")
            output.append({"sample_id": row["sample_id"], "dataset": dataset, "method": method,
                           "retention_ratio": ratio, "emotion_code": letter, "emotion": emotion,
                           "ground_truth": truth, "video_only_probability": p_only,
                           "video_utterance_probability": p_both,
                           "probability_delta_with_utterance": p_both - p_only,
                           "video_only_ground_truth_probability": gt_only,
                           "video_utterance_ground_truth_probability": gt_both,
                           "ground_truth_probability_delta": gt_both - gt_only if gt_only != "" else "",
                           "video_only_prediction": pred_only,
                           "video_utterance_prediction": pred_both,
                           "full_visual_tokens": full, "retained_visual_tokens": retained})
        del only_inputs, both_inputs
    return output


def read_results(path, rows, dataset, method):
    if not path.is_file():
        return [], set()
    import csv
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != FIELDS:
            raise ValueError(f"{path}: result columns differ")
        saved = list(reader)
    expected = {row["sample_id"] for row in rows}
    groups = {}
    for item in saved:
        if item["dataset"] != dataset or item["method"] != method:
            raise ValueError(f"{path}: wrong dataset or method")
        groups.setdefault(item["sample_id"], []).append(item)
    labels = set(letter_map(dataset))
    if not set(groups) <= expected or any(len(group) != len(labels) or
        {item["emotion_code"] for item in group} != labels for group in groups.values()):
        raise ValueError(f"{path}: incomplete or mismatched saved clip")
    return saved, set(groups)


def summarize(rows, dataset, method, ratio, total):
    if len({row["sample_id"] for row in rows}) != total:
        raise ValueError("Cannot summarize incomplete test coverage")
    def mean(values):
        return sum(values) / len(values) if values else None
    groups = {}
    for name in letter_map(dataset).values():
        group = [row for row in rows if row["emotion"] == name]
        gt = [row for row in group if row["ground_truth_probability_delta"] != ""]
        positive = [row for row in gt if int(row["ground_truth"])]
        groups[name] = {"ground_truth_positive_count": sum(int(row["ground_truth"]) for row in group),
                        "mean_probability_delta_with_utterance": mean([
                            float(row["probability_delta_with_utterance"]) for row in group]),
                        "mean_ground_truth_probability_delta": mean([
                            float(row["ground_truth_probability_delta"]) for row in gt]),
                        "mean_positive_ground_truth_probability_delta": mean([
                            float(row["ground_truth_probability_delta"]) for row in positive])}
    gt = [row for row in rows if row["ground_truth_probability_delta"] != ""]
    positive = [row for row in gt if int(row["ground_truth"])]
    return {"dataset": dataset, "method": method, "retention_ratio": ratio,
            "selection_basis": "highest previously observed MOSEI test mean weighted accuracy or MELD test accuracy below 50%; exploratory",
            "completed_clips": total, "rows": len(rows),
            "mean_ground_truth_probability_delta": mean([
                float(row["ground_truth_probability_delta"]) for row in gt]),
            "mean_positive_ground_truth_probability_delta": mean([
                float(row["ground_truth_probability_delta"]) for row in positive]),
            "fraction_ground_truth_probability_increased": mean([
                float(float(row["ground_truth_probability_delta"]) > 0) for row in gt]),
            "per_emotion": groups}


def plot_summary(rows, report, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = list(report["per_emotion"])
    values = [report["per_emotion"][name]["mean_positive_ground_truth_probability_delta"]
              for name in names]
    deltas = [float(row["ground_truth_probability_delta"]) for row in rows
              if int(row["ground_truth"]) and row["ground_truth_probability_delta"] != ""]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    axes[0].bar(names, [value if value is not None else 0 for value in values])
    axes[0].tick_params(axis="x", rotation=45)
    axes[0].axhline(0, color="black", linewidth=.8)
    axes[0].set_ylabel("Mean ΔP(ground truth)")
    axes[0].set_title("With utterance − visual only")
    axes[1].hist(deltas, bins=40)
    axes[1].axvline(0, color="black", linewidth=.8)
    axes[1].set_xlabel("ΔP(ground truth)")
    axes[1].set_ylabel("Positive ground-truth labels / clips")
    axes[1].set_title("Paired test-set changes")
    fig.suptitle(f"{report['dataset'].upper()} {report['method']} at {report['retention_ratio']:.0%}")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, choices=tuple(RATIOS))
    parser.add_argument("--method", default="all", choices=("all", *RATIOS["mosei"]))
    parser.add_argument("--root", type=Path, default=REPO.parents[1])
    parser.add_argument("--model-dir", type=Path)
    parser.add_argument("--dataset-root", type=Path)
    parser.add_argument("--aligned-csv", type=Path, help="MOSEI six-label aligned CSV")
    parser.add_argument("--decoded-cache-dir", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument("--checkpoint-every", type=int, default=10)
    parser.add_argument("--limit", type=int, help="Development only: first N selected test clips")
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    if args.checkpoint_every < 1 or args.random_seed < 0 or (args.limit is not None and args.limit < 1):
        parser.error("Checkpoint interval and limit must be positive; random seed nonnegative")
    root = args.root.expanduser().resolve()
    data_root = (args.dataset_root or root / "datasets" / args.dataset).expanduser().resolve()
    model_dir = (args.model_dir or root / "models" / "qwen").expanduser().resolve()
    cache_dir = resolve_cache_frames((args.decoded_cache_dir or
        data_root / "decoded_qwen_6fps_224_v1").expanduser().resolve())
    output = (args.output_dir or root / "results" /
        f"qwen2_5_vl_7b_{args.dataset}_retained_visual_utterance_lm_head_test").expanduser().resolve()
    protocol = json.loads(Path(__file__).with_name(f"{args.dataset}_video_protocol.json").read_text())
    protocol["prompt_style"] = "emotion_multilabel_task" if args.dataset == "mosei" else "meld_emotion_task"
    verify_qwen_environment(curve=True)
    verify_model(model_dir, "qwen")
    if args.dataset == "mosei":
        data_root = Path(verify_mosei(data_root)["root"])
        label_csv = (args.aligned_csv or data_root / "csv" / "mosei_aligned.csv").expanduser().resolve()
        fields, all_rows, selection = cached_multilabel_rows(label_csv, cache_dir, data_root / "Raw", protocol)
        rows = [dict(row, _video_path=str(data_root / "Raw" / row["video_id"] /
                              f"{row['clip_id']}.mp4")) for row in all_rows if row["split"] == "test"]
    else:
        import subprocess
        import sys
        data_root = Path(verify_meld(data_root)["root"])
        index_dir = output / "indexes_all"
        index_dir.mkdir(parents=True, exist_ok=True)
        subprocess.run([sys.executable, "-u", str(REPO / "scripts/data/prepare_meld.py"),
                        "--meld-root", str(data_root), "--output-dir", str(index_dir)], check=True)
        selected, selection = selected_cached_rows(index_dir, cache_dir, protocol)
        rows = selected["test"]
        fields = list(rows[0])
        label_csv = None
    if args.limit is not None:
        rows = rows[:args.limit]
    if not rows:
        raise ValueError("No valid cached test clips selected")
    print(f"{args.dataset.upper()} selected test clips: {len(rows)}; "
          f"cache selection: {selection['selected']}; output: {output}", flush=True)
    methods = tuple(RATIOS[args.dataset]) if args.method == "all" else (args.method,)
    print(f"Methods/ratios: {[(method, RATIOS[args.dataset][method]) for method in methods]}", flush=True)
    if args.verify_only:
        return 0
    import torch
    import transformers
    if transformers.__version__ != "4.57.1":
        raise RuntimeError("This experiment requires transformers==4.57.1")
    require_cuda_gpu(torch)
    os.environ["HF_HUB_OFFLINE"] = "1"
    model = transformers.Qwen2_5_VLForConditionalGeneration.from_pretrained(
        str(model_dir), dtype=torch.bfloat16, device_map={"": 0}, local_files_only=True,
        attn_implementation="sdpa").eval()
    processor = transformers.AutoProcessor.from_pretrained(str(model_dir), local_files_only=True)
    choices = ("Y", "N") if args.dataset == "mosei" else tuple(letter_map(args.dataset))
    token_ids = single_token_ids(processor.tokenizer, choices)
    selected_rows = [{key: row[key] for key in fields} for row in rows]
    for method in methods:
        ratio = RATIOS[args.dataset][method]
        folder = output / method
        config = {"version": VERSION, "dataset": args.dataset, "method": method,
                  "retention_ratio": ratio, "model_dir": str(model_dir),
                  "model_config_sha256": hashlib.sha256((model_dir / "config.json").read_bytes()).hexdigest(),
                  "dataset_root": str(data_root), "label_csv": str(label_csv) if label_csv else None,
                  "cache_dir": str(cache_dir), "protocol": protocol,
                  "selected_sha256": digest_rows(selected_rows), "selected_count": len(rows),
                  "random_seed": args.random_seed if method == "random" else None,
                  "candidate_tokens": dict(zip(choices, token_ids)),
                  "answer_protocol": "first LM-head token; candidate-normalized softmax; MOSEI has independent Y/N queries",
                  "conditions": ["retained_video_only", "retained_video_with_utterance"],
                  "selection_basis": "test metric, exploratory"}
        config_path = folder / "run_config.json"
        if config_path.is_file() and json.loads(config_path.read_text()) != config:
            raise ValueError(f"{config_path}: saved configuration differs")
        atomic_json(config_path, config)
        atomic_csv(folder / "selected_test.csv", selected_rows, fields)
        result_path = folder / "per_emotion_probabilities.csv"
        results, done = read_results(result_path, rows, args.dataset, method)
        print(f"{method} {ratio:.0%}: resuming {len(done)}/{len(rows)}", flush=True)
        for index, row in enumerate(rows, 1):
            if row["sample_id"] in done:
                continue
            try:
                scored = score_clip(row, args.dataset, method, ratio, model, processor,
                                    cache_dir, protocol, token_ids, torch, args.random_seed)
            except Exception:
                atomic_csv(result_path, results, FIELDS)
                raise
            results.extend(scored)
            done.add(row["sample_id"])
            if len(done) % args.checkpoint_every == 0:
                atomic_csv(result_path, results, FIELDS)
            print(f"{method} [{index}/{len(rows)}] {row['sample_id']}: "
                  f"retained {scored[0]['retained_visual_tokens']}/{scored[0]['full_visual_tokens']}", flush=True)
        atomic_csv(result_path, results, FIELDS)
        report = summarize(results, args.dataset, method, ratio, len(rows))
        atomic_json(folder / "summary.json", report)
        plot_summary(results, report, folder / "ground_truth_probability_change.png")
        print(f"Complete: {folder}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
