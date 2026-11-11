#!/usr/bin/env python3
"""Compare test-clip LM-head emotion probabilities across four input conditions.

MOSEI uses its six multilabel emotion scores (>0 means present). MELD uses its
seven-class label. Full and no-visual share a video prompt; utterance-only
does not invoke the vision encoder; video-only never includes the utterance.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from cluster.meld_cache_subset import selected_cached_rows  # noqa: E402
from cluster.mosei_cache_subset import cached_multilabel_rows, resolve_cache_frames  # noqa: E402
from cluster.mosei_decoded_cache import identity, load_cached  # noqa: E402
from cluster.preflight import verify_meld, verify_model, verify_mosei  # noqa: E402
from cluster.qwen_no_visual_tokens import NoVisualTokensHook  # noqa: E402
from cluster.verify_qwen_environment import require_cuda_gpu, verify_qwen_environment  # noqa: E402
from scripts.data.prepare_meld import LABELS as MELD_LABELS  # noqa: E402

MOSEI_EMOTIONS = ("anger", "disgust", "fear", "happy", "sad", "surprise")
MELD_EMOTIONS = tuple(MELD_LABELS)
RESULT_FIELDS = ("sample_id", "dataset", "emotion_code", "emotion", "ground_truth",
                 "full_probability", "no_visual_probability", "delta_full_minus_no_visual",
                 "full_ground_truth_probability", "no_visual_ground_truth_probability",
                 "ground_truth_probability_delta", "full_prediction", "no_visual_prediction",
                 "visual_tokens", "utterance_only_probability",
                 "delta_full_minus_utterance_only", "delta_no_visual_minus_utterance_only",
                 "utterance_only_ground_truth_probability",
                 "ground_truth_probability_delta_full_minus_utterance_only",
                 "ground_truth_probability_delta_no_visual_minus_utterance_only",
                 "utterance_only_prediction", "video_only_probability",
                 "delta_full_minus_video_only", "delta_no_visual_minus_video_only",
                 "delta_video_only_minus_utterance_only",
                 "video_only_ground_truth_probability",
                 "ground_truth_probability_delta_full_minus_video_only",
                 "ground_truth_probability_delta_no_visual_minus_video_only",
                 "ground_truth_probability_delta_video_only_minus_utterance_only",
                 "video_only_prediction")
PROMPT_VERSION = "lm_head_visual_ablation_v3_video_only"


def atomic_csv(path: Path, rows: list[dict], fields: list[str] | tuple[str, ...]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    try:
        with temp.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
        temp.replace(path)
    finally:
        temp.unlink(missing_ok=True)


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    try:
        temp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        temp.replace(path)
    finally:
        temp.unlink(missing_ok=True)


def digest_rows(rows: list[dict]) -> str:
    data = json.dumps(rows, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(data).hexdigest()


def letter_map(dataset: str) -> dict[str, str]:
    names = MOSEI_EMOTIONS if dataset == "mosei" else MELD_EMOTIONS
    return {chr(65 + index): name for index, name in enumerate(names)}


def single_token_ids(tokenizer, choices: tuple[str, ...]) -> list[int]:
    ids = [tokenizer.encode(choice, add_special_tokens=False) for choice in choices]
    if any(len(value) != 1 for value in ids) or len({value[0] for value in ids}) != len(ids):
        raise ValueError(f"Expected distinct single tokens for {choices}: {ids}")
    return [value[0] for value in ids]


def prompt_for(dataset: str, utterance: str, code: str | None = None,
               modality: str = "video") -> str:
    if modality not in {"video", "utterance_only", "video_only"}:
        raise ValueError(f"Unknown prompt modality: {modality}")
    evidence = {"video": "the video and utterance",
                "utterance_only": "the utterance alone",
                "video_only": "the video alone"}[modality]
    utterance_line = "" if modality == "video_only" else f"\nUtterance: {utterance}"
    if dataset == "meld":
        choices = ", ".join(f"{key}={value}" for key, value in letter_map(dataset).items())
        return (f"Classify the speaker's emotion using {evidence}. "
                f"Choose exactly one code: {choices}.{utterance_line}\n"
                "Answer with exactly one capital letter, A through G.")
    assert code is not None
    emotion = letter_map(dataset)[code]
    return (f"Assess whether emotion {code}={emotion} is present in the speaker, using "
            f"{evidence}.{utterance_line}\n"
            "Answer exactly Y if present, or N if absent.")


def messages_for(video_path: Path, protocol: dict, prompt: str) -> list[dict]:
    return [{"role": "user", "content": [
        {"type": "video", "video": video_path.as_uri(), "fps": protocol["fps"],
         "min_frames": protocol["min_frames"], "max_frames": protocol["max_frames"],
         "resized_height": protocol["frame_size"],
         "resized_width": protocol["frame_size"]},
        {"type": "text", "text": prompt},
    ]}]


def text_messages_for(prompt: str) -> list[dict]:
    return [{"role": "user", "content": [{"type": "text", "text": prompt}]}]


def score_candidates(model, inputs, candidate_ids: list[int], torch,
                     remove_video: bool = False) -> list[float]:
    with torch.inference_mode():
        if remove_video:
            with NoVisualTokensHook(model, inputs["input_ids"]):
                output = model.model(**inputs, use_cache=False, return_dict=True)
        else:
            output = model.model(**inputs, use_cache=False, return_dict=True)
        logits = model.lm_head(output.last_hidden_state[0, -1]).float()
        probabilities = torch.softmax(logits[candidate_ids], dim=0).cpu().tolist()
        del output, logits
    return probabilities


def probability_pair(model, inputs, candidate_ids: list[int], torch) -> tuple[list[float], list[float], int]:
    """Return candidate-normalized probabilities for both conditions."""
    input_ids = inputs["input_ids"]
    video_tokens = int(input_ids.eq(model.config.video_token_id).sum())
    full = score_candidates(model, inputs, candidate_ids, torch)
    absent = score_candidates(model, inputs, candidate_ids, torch, remove_video=True)
    return full, absent, video_tokens


def utterance_only_probability(model, processor, dataset: str, utterance: str,
                               code: str | None, candidate_ids: list[int], torch) -> list[float]:
    prompt = prompt_for(dataset, utterance, code, modality="utterance_only")
    text = processor.apply_chat_template(text_messages_for(prompt), tokenize=False,
                                         add_generation_prompt=True)
    inputs = processor(text=[text], return_tensors="pt").to(model.device)
    if any(key in inputs for key in ("pixel_values", "pixel_values_videos", "video_grid_thw")):
        raise ValueError("Utterance-only preprocessing unexpectedly produced visual inputs")
    if int(inputs["input_ids"].eq(model.config.video_token_id).sum()):
        raise ValueError("Utterance-only prompt unexpectedly contains video tokens")
    return score_candidates(model, inputs, candidate_ids, torch)


def video_only_probability(model, processor, dataset: str, code: str | None,
                           video_path: Path, protocol: dict, images, video_tensors,
                           video_metadata, video_kwargs: dict,
                           candidate_ids: list[int], torch) -> list[float]:
    # No utterance is accepted here, so the prompt cannot contain a clip transcript.
    prompt = prompt_for(dataset, "", code, modality="video_only")
    text = processor.apply_chat_template(
        messages_for(video_path, protocol, prompt), tokenize=False,
        add_generation_prompt=True)
    inputs = processor(text=[text], images=images, videos=list(video_tensors),
                       video_metadata=list(video_metadata), return_tensors="pt",
                       do_resize=False, **video_kwargs).to(model.device)
    if not int(inputs["input_ids"].eq(model.config.video_token_id).sum()):
        raise ValueError("Video-only prompt did not produce video tokens")
    return score_candidates(model, inputs, candidate_ids, torch)


def score_clip(row: dict, dataset: str, model, processor, cache_dir: Path,
               protocol: dict, token_ids: list[int], torch) -> list[dict]:
    sample_id = row["sample_id"]
    utterance = " ".join(row["text"].split()) if dataset == "mosei" else row["utterance"]
    video_path = (Path(row["video_path"]) if dataset == "meld" else
                  Path(row["_video_path"]))
    cached = load_cached(cache_dir, identity(sample_id, video_path, protocol))
    if cached is None:
        raise RuntimeError(f"Cached frames disappeared for {sample_id}")
    images, videos, video_kwargs = cached
    video_tensors, video_metadata = zip(*videos)
    labels = letter_map(dataset)
    codes = tuple(labels) if dataset == "mosei" else (None,)
    output_rows = []
    for code in codes:
        prompt = prompt_for(dataset, utterance, code)
        messages = messages_for(video_path, protocol, prompt)
        text = processor.apply_chat_template(messages, tokenize=False,
                                             add_generation_prompt=True)
        inputs = processor(text=[text], images=images, videos=list(video_tensors),
                           video_metadata=list(video_metadata), return_tensors="pt",
                           do_resize=False, **video_kwargs).to(model.device)
        full, absent, visual_tokens = probability_pair(model, inputs, token_ids, torch)
        utterance_only = utterance_only_probability(
            model, processor, dataset, utterance, code, token_ids, torch)
        video_only = video_only_probability(
            model, processor, dataset, code, video_path, protocol, images,
            video_tensors, video_metadata, video_kwargs, token_ids, torch)
        if dataset == "meld":
            truth = row["emotion"].lower()
            if truth not in labels.values():
                raise ValueError(f"Unknown MELD label for {sample_id}: {truth}")
            full_prediction = labels[chr(65 + max(range(7), key=full.__getitem__))]
            absent_prediction = labels[chr(65 + max(range(7), key=absent.__getitem__))]
            text_prediction = labels[chr(65 + max(range(7), key=utterance_only.__getitem__))]
            video_prediction = labels[chr(65 + max(range(7), key=video_only.__getitem__))]
            for index, (letter, emotion) in enumerate(labels.items()):
                gt = int(emotion == truth)
                output_rows.append({
                    "sample_id": sample_id, "dataset": dataset, "emotion_code": letter,
                    "emotion": emotion, "ground_truth": gt,
                    "full_probability": full[index], "no_visual_probability": absent[index],
                    "utterance_only_probability": utterance_only[index],
                    "video_only_probability": video_only[index],
                    "delta_full_minus_no_visual": full[index] - absent[index],
                    "delta_full_minus_utterance_only": full[index] - utterance_only[index],
                    "delta_no_visual_minus_utterance_only": absent[index] - utterance_only[index],
                    "delta_full_minus_video_only": full[index] - video_only[index],
                    "delta_no_visual_minus_video_only": absent[index] - video_only[index],
                    "delta_video_only_minus_utterance_only": video_only[index] - utterance_only[index],
                    "full_ground_truth_probability": full[index] if gt else "",
                    "no_visual_ground_truth_probability": absent[index] if gt else "",
                    "utterance_only_ground_truth_probability": utterance_only[index] if gt else "",
                    "video_only_ground_truth_probability": video_only[index] if gt else "",
                    "ground_truth_probability_delta": full[index] - absent[index] if gt else "",
                    "ground_truth_probability_delta_full_minus_utterance_only": (
                        full[index] - utterance_only[index] if gt else ""),
                    "ground_truth_probability_delta_no_visual_minus_utterance_only": (
                        absent[index] - utterance_only[index] if gt else ""),
                    "ground_truth_probability_delta_full_minus_video_only": (
                        full[index] - video_only[index] if gt else ""),
                    "ground_truth_probability_delta_no_visual_minus_video_only": (
                        absent[index] - video_only[index] if gt else ""),
                    "ground_truth_probability_delta_video_only_minus_utterance_only": (
                        video_only[index] - utterance_only[index] if gt else ""),
                    "full_prediction": full_prediction, "no_visual_prediction": absent_prediction,
                    "utterance_only_prediction": text_prediction,
                    "video_only_prediction": video_prediction,
                    "visual_tokens": visual_tokens,
                })
        else:
            assert code is not None
            emotion = labels[code]
            gt = int(float(row[emotion]) > 0)
            full_yes, absent_yes = full[0], absent[0]
            text_yes = utterance_only[0]
            video_yes = video_only[0]
            output_rows.append({
                "sample_id": sample_id, "dataset": dataset, "emotion_code": code,
                "emotion": emotion, "ground_truth": gt,
                "full_probability": full_yes, "no_visual_probability": absent_yes,
                "utterance_only_probability": text_yes,
                "video_only_probability": video_yes,
                "delta_full_minus_no_visual": full_yes - absent_yes,
                "delta_full_minus_utterance_only": full_yes - text_yes,
                "delta_no_visual_minus_utterance_only": absent_yes - text_yes,
                "delta_full_minus_video_only": full_yes - video_yes,
                "delta_no_visual_minus_video_only": absent_yes - video_yes,
                "delta_video_only_minus_utterance_only": video_yes - text_yes,
                "full_ground_truth_probability": full_yes if gt else 1 - full_yes,
                "no_visual_ground_truth_probability": absent_yes if gt else 1 - absent_yes,
                "utterance_only_ground_truth_probability": text_yes if gt else 1 - text_yes,
                "video_only_ground_truth_probability": video_yes if gt else 1 - video_yes,
                "ground_truth_probability_delta": (full_yes - absent_yes) if gt else (absent_yes - full_yes),
                "ground_truth_probability_delta_full_minus_utterance_only": (
                    full_yes - text_yes if gt else text_yes - full_yes),
                "ground_truth_probability_delta_no_visual_minus_utterance_only": (
                    absent_yes - text_yes if gt else text_yes - absent_yes),
                "ground_truth_probability_delta_full_minus_video_only": (
                    full_yes - video_yes if gt else video_yes - full_yes),
                "ground_truth_probability_delta_no_visual_minus_video_only": (
                    absent_yes - video_yes if gt else video_yes - absent_yes),
                "ground_truth_probability_delta_video_only_minus_utterance_only": (
                    video_yes - text_yes if gt else text_yes - video_yes),
                "full_prediction": int(full_yes >= 0.5),
                "no_visual_prediction": int(absent_yes >= 0.5),
                "utterance_only_prediction": int(text_yes >= 0.5),
                "video_only_prediction": int(video_yes >= 0.5),
                "visual_tokens": visual_tokens,
            })
        del inputs
    return output_rows


def summarize(rows: list[dict], dataset: str, total_clips: int) -> dict:
    if not rows:
        raise ValueError("No scored test clips")
    per_class = {}
    for code, emotion in letter_map(dataset).items():
        group = [row for row in rows if row["emotion_code"] == code]
        positive = [row for row in group if int(row["ground_truth"])]
        def mean(values):
            return sum(values) / len(values) if values else None
        per_class[emotion] = {
            "code": code, "positive_count": len(positive),
            "mean_full_probability": mean([float(row["full_probability"]) for row in group]),
            "mean_no_visual_probability": mean([float(row["no_visual_probability"]) for row in group]),
            "mean_utterance_only_probability": mean([
                float(row["utterance_only_probability"]) for row in group]),
            "mean_video_only_probability": mean([
                float(row["video_only_probability"]) for row in group]),
            "mean_probability_change": mean([float(row["delta_full_minus_no_visual"]) for row in group]),
            "mean_probability_change_full_minus_utterance_only": mean([
                float(row["delta_full_minus_utterance_only"]) for row in group]),
            "mean_probability_change_no_visual_minus_utterance_only": mean([
                float(row["delta_no_visual_minus_utterance_only"]) for row in group]),
            "mean_probability_change_full_minus_video_only": mean([
                float(row["delta_full_minus_video_only"]) for row in group]),
            "mean_probability_change_no_visual_minus_video_only": mean([
                float(row["delta_no_visual_minus_video_only"]) for row in group]),
            "mean_probability_change_video_only_minus_utterance_only": mean([
                float(row["delta_video_only_minus_utterance_only"]) for row in group]),
            "mean_positive_ground_truth_change": mean([
                float(row["ground_truth_probability_delta"]) for row in positive]),
            "mean_positive_ground_truth_change_full_minus_utterance_only": mean([
                float(row["ground_truth_probability_delta_full_minus_utterance_only"])
                for row in positive]),
            "mean_positive_ground_truth_change_no_visual_minus_utterance_only": mean([
                float(row["ground_truth_probability_delta_no_visual_minus_utterance_only"])
                for row in positive]),
            "mean_positive_ground_truth_change_full_minus_video_only": mean([
                float(row["ground_truth_probability_delta_full_minus_video_only"])
                for row in positive]),
            "mean_positive_ground_truth_change_no_visual_minus_video_only": mean([
                float(row["ground_truth_probability_delta_no_visual_minus_video_only"])
                for row in positive]),
            "mean_positive_ground_truth_change_video_only_minus_utterance_only": mean([
                float(row["ground_truth_probability_delta_video_only_minus_utterance_only"])
                for row in positive]),
        }
    clips = len({row["sample_id"] for row in rows})
    gt_rows = [row for row in rows if int(row["ground_truth"])]
    return {"dataset": dataset, "split": "test", "selected_clips": total_clips,
            "completed_clips": clips, "complete": clips == total_clips,
            "probability_definition": ("softmax over seven A-G code logits" if dataset == "meld"
                                       else "softmax over Y/N logits for each independent emotion query"),
            "mean_positive_ground_truth_probability_change": (sum(
                float(row["ground_truth_probability_delta"]) for row in gt_rows) / len(gt_rows)
                if gt_rows else None),
            "mean_positive_ground_truth_probability_change_full_minus_utterance_only": (sum(
                float(row["ground_truth_probability_delta_full_minus_utterance_only"])
                for row in gt_rows) / len(gt_rows) if gt_rows else None),
            "mean_positive_ground_truth_probability_change_no_visual_minus_utterance_only": (sum(
                float(row["ground_truth_probability_delta_no_visual_minus_utterance_only"])
                for row in gt_rows) / len(gt_rows) if gt_rows else None),
            "mean_positive_ground_truth_probability_change_full_minus_video_only": (sum(
                float(row["ground_truth_probability_delta_full_minus_video_only"])
                for row in gt_rows) / len(gt_rows) if gt_rows else None),
            "mean_positive_ground_truth_probability_change_no_visual_minus_video_only": (sum(
                float(row["ground_truth_probability_delta_no_visual_minus_video_only"])
                for row in gt_rows) / len(gt_rows) if gt_rows else None),
            "mean_positive_ground_truth_probability_change_video_only_minus_utterance_only": (sum(
                float(row["ground_truth_probability_delta_video_only_minus_utterance_only"])
                for row in gt_rows) / len(gt_rows) if gt_rows else None),
            "per_class": per_class}


def plot_summary(rows: list[dict], report: dict, path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    classes = list(report["per_class"])
    video_values = [report["per_class"][name]["mean_positive_ground_truth_change"]
                    for name in classes]
    text_values = [report["per_class"][name][
        "mean_positive_ground_truth_change_full_minus_utterance_only"] for name in classes]
    visual_values = [report["per_class"][name][
        "mean_positive_ground_truth_change_full_minus_video_only"] for name in classes]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    positions = list(range(len(classes)))
    axes[0].bar([x - 0.26 for x in positions],
                [float(value) if value is not None else math.nan for value in video_values],
                width=0.26, label="Full − no visual tokens")
    axes[0].bar(positions,
                [float(value) if value is not None else math.nan for value in text_values],
                width=0.26, label="Full − utterance only")
    axes[0].bar([x + 0.26 for x in positions],
                [float(value) if value is not None else math.nan for value in visual_values],
                width=0.26, label="Full − video only")
    axes[0].axhline(0, color="black", linewidth=0.8)
    axes[0].set_xticks(positions, classes, rotation=40)
    axes[0].set_ylabel("Mean change in P(ground truth)")
    axes[0].set_title("Positive ground-truth labels by emotion")
    axes[0].legend(fontsize=8)
    gt_video = [float(row["ground_truth_probability_delta"]) for row in rows
                if int(row["ground_truth"])]
    gt_text = [float(row["ground_truth_probability_delta_full_minus_utterance_only"])
               for row in rows if int(row["ground_truth"])]
    gt_visual = [float(row["ground_truth_probability_delta_full_minus_video_only"])
                 for row in rows if int(row["ground_truth"])]
    axes[1].hist([gt_video, gt_text, gt_visual], bins=40,
                 label=["Full − no visual tokens", "Full − utterance only",
                        "Full − video only"])
    axes[1].axvline(0, color="black", linewidth=0.8)
    axes[1].set_xlabel("Change in P(ground truth)")
    axes[1].set_ylabel("Positive labels")
    axes[1].set_title("Test-set probability changes")
    axes[1].legend(fontsize=8)
    fig.suptitle(f"{report['dataset'].upper()} LM-head modality comparison")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def read_results(path: Path, rows: list[dict], dataset: str) -> tuple[list[dict], set[str]]:
    if not path.is_file():
        return [], set()
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != RESULT_FIELDS:
            raise ValueError(f"{path}: result columns differ from current experiment")
        saved = list(reader)
    expected = {row["sample_id"] for row in rows}
    groups = {}
    for row in saved:
        groups.setdefault(row["sample_id"], []).append(row)
    labels = set(letter_map(dataset))
    if not set(groups) <= expected or any(
        len(group) != len(labels) or {item["emotion_code"] for item in group} != labels
        for group in groups.values()
    ):
        raise ValueError(f"{path}: incomplete or mismatched saved clip")
    return saved, set(groups)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=("mosei", "meld"), required=True)
    parser.add_argument("--root", type=Path, default=REPO.parents[1])
    parser.add_argument("--model-dir", type=Path)
    parser.add_argument("--dataset-root", type=Path)
    parser.add_argument("--decoded-cache-dir", type=Path)
    parser.add_argument("--aligned-csv", type=Path, help="MOSEI multilabel CSV")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--checkpoint-every", type=int, default=10)
    parser.add_argument("--limit", type=int, help="Development only: first N selected test clips")
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    if args.checkpoint_every < 1 or (args.limit is not None and args.limit < 1):
        parser.error("Checkpoint interval and limit must be positive")
    root = args.root.expanduser().resolve()
    data_root = (args.dataset_root or root / "datasets" / args.dataset).expanduser().resolve()
    model_dir = (args.model_dir or root / "models" / "qwen").expanduser().resolve()
    cache_dir = resolve_cache_frames((args.decoded_cache_dir or
        data_root / "decoded_qwen_6fps_224_v1").expanduser().resolve())
    output = (args.output_dir or root / "results" /
        f"qwen2_5_vl_7b_{args.dataset}_lm_head_four_conditions_test").expanduser().resolve()
    protocol_file = Path(__file__).with_name(f"{args.dataset}_video_protocol.json")
    protocol = json.loads(protocol_file.read_text())
    protocol["prompt_style"] = "emotion_multilabel_task" if args.dataset == "mosei" else "meld_emotion_task"
    verify_qwen_environment(curve=True)
    verify_model(model_dir, "qwen")
    if args.dataset == "mosei":
        data_root = Path(verify_mosei(data_root)["root"])
        label_csv = (args.aligned_csv or data_root / "csv" / "mosei_aligned.csv").expanduser().resolve()
        fields, all_rows, selection = cached_multilabel_rows(
            label_csv, cache_dir, data_root / "Raw", protocol)
        rows = [dict(row, _video_path=str(data_root / "Raw" / row["video_id"] /
                              f"{row['clip_id']}.mp4")) for row in all_rows if row["split"] == "test"]
        selected_fields = fields
    else:
        data_root = Path(verify_meld(data_root)["root"])
        index_dir = output / "indexes_all"
        index_dir.mkdir(parents=True, exist_ok=True)
        subprocess.run([sys.executable, "-u", str(REPO / "scripts/data/prepare_meld.py"),
                        "--meld-root", str(data_root), "--output-dir", str(index_dir)], check=True)
        selected, selection = selected_cached_rows(index_dir, cache_dir, protocol)
        rows = selected["test"]
        selected_fields = list(rows[0])
        label_csv = None
    if args.limit is not None:
        rows = rows[:args.limit]
    if not rows:
        raise ValueError("No valid cached test clips selected")
    print(f"{args.dataset.upper()} selected test clips: {len(rows)}; "
          f"cache selection: {selection['selected']}; output: {output}", flush=True)
    print(f"Emotion codes: {letter_map(args.dataset)}", flush=True)
    if args.verify_only:
        return 0

    import torch
    import transformers
    if transformers.__version__ != "4.57.1":
        raise RuntimeError("This experiment requires transformers==4.57.1")
    require_cuda_gpu(torch)
    os.environ["HF_HUB_OFFLINE"] = "1"
    model = transformers.Qwen2_5_VLForConditionalGeneration.from_pretrained(
        str(model_dir), dtype=torch.bfloat16, device_map={"": 0},
        local_files_only=True, attn_implementation="sdpa").eval()
    processor = transformers.AutoProcessor.from_pretrained(str(model_dir), local_files_only=True)
    choices = ("Y", "N") if args.dataset == "mosei" else tuple(letter_map(args.dataset))
    token_ids = single_token_ids(processor.tokenizer, choices)
    selected_rows = [{key: row[key] for key in selected_fields} for row in rows]
    config = {"experiment": PROMPT_VERSION, "dataset": args.dataset, "model_dir": str(model_dir),
              "model_config_sha256": hashlib.sha256((model_dir / "config.json").read_bytes()).hexdigest(),
              "dataset_root": str(data_root), "label_csv": str(label_csv) if label_csv else None,
              "cache_dir": str(cache_dir), "protocol": protocol, "selected_sha256": digest_rows(selected_rows),
              "selected_count": len(rows), "codes": letter_map(args.dataset),
              "candidate_tokens": dict(zip(choices, token_ids)), "answer_protocol":
              "first LM-head token; candidate-normalized softmax; independent Y/N queries for MOSEI",
              "conditions": ["full", "no_visual_tokens", "utterance_only", "video_only"]}
    config_path = output / "run_config.json"
    if config_path.is_file() and json.loads(config_path.read_text()) != config:
        raise ValueError(f"{config_path}: saved run configuration differs; choose another output directory")
    atomic_json(config_path, config)
    atomic_csv(output / "selected_test.csv", selected_rows, selected_fields)
    results_path = output / "per_emotion_probabilities.csv"
    results, done = read_results(results_path, rows, args.dataset)
    print(f"Resuming {len(done)}/{len(rows)} test clips", flush=True)
    for index, row in enumerate(rows, start=1):
        if row["sample_id"] in done:
            continue
        try:
            scored = score_clip(row, args.dataset, model, processor, cache_dir, protocol, token_ids, torch)
        except Exception:
            atomic_csv(results_path, results, RESULT_FIELDS)
            raise
        results.extend(scored)
        done.add(row["sample_id"])
        if len(done) % args.checkpoint_every == 0:
            atomic_csv(results_path, results, RESULT_FIELDS)
        print(f"[{index}/{len(rows)}] {row['sample_id']}: "
              f"{len(done)} completed; visual tokens={scored[0]['visual_tokens']}", flush=True)
    atomic_csv(results_path, results, RESULT_FIELDS)
    report = summarize(results, args.dataset, len(rows))
    atomic_json(output / "summary.json", report)
    plot_summary(results, report, output / "lm_head_probability_comparison.png")
    print(f"Complete: {output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
