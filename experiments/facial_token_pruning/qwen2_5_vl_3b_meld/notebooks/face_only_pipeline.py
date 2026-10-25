"""Kaggle-ready face-only Qwen visual-token experiment.

Paste this file into a Kaggle cell, or run the sibling notebook. Keep the
installed benchmark-code dataset and MELD dataset attached to the notebook.
"""

from __future__ import annotations

import csv
import json
import os
import random
import sys
from collections import defaultdict
from pathlib import Path

os.environ.setdefault("FORCE_QWENVL_VIDEO_READER", "decord")
os.environ.setdefault("DEEPFACE_BACKEND_ENGINE", "onnx")
os.environ.setdefault("VIDEO_MAX_PIXELS", str(224**2 * 64))

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from deepface import DeepFace
from IPython.display import display
from qwen_vl_utils import process_vision_info
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

REPO = Path("/kaggle/input/datasets/pushkarsingh2005/benchmark-code")
INDEX_CSV = REPO / "data/indexes/meld_dev_index.csv"
# Set this to the exact frozen MLP checkpoint used in your previous experiment.
# If None, the code searches attached Kaggle inputs and requires one clear match.
CHECKPOINT = None
MODEL_ID = "Qwen/Qwen2.5-VL-3B-Instruct"
OUTPUT_DIR = Path("/kaggle/working/face_only_tokens")
START = 0
LIMIT = 1  # Increase for paired dev-set metrics; first validate one sample.
FPS, MIN_FRAMES, MAX_FRAMES, FRAME_SIZE = 6.0, 4, 64, 224
PROMPT_STYLE = "emotion_task"
POOLING = "last"  # Must match the checkpoint's training extraction.
OVERLAP_FRACTION = 0.10
DETECTOR = "retinaface"
CONTROL_SEEDS = (0, 1, 2, 3, 4)
SHOW_FIRST_MAP = True

sys.path.insert(0, str(REPO))
from scripts.classification.evaluate_mlp import EmotionMLP, standardize_with_checkpoint
from scripts.extraction.extract_qwen_shared_embeddings import (
    fix_video_kwargs, make_message, pool_hidden_states,
)
from scripts.pruning.divprune import select_diverse_token_indices


def choose_checkpoint():
    if CHECKPOINT is not None:
        path = Path(CHECKPOINT)
        if not path.is_file():
            raise FileNotFoundError(path)
        return path
    paths = (list(Path("/kaggle/input").rglob("best_mlp.pt"))
             + list(Path("/kaggle/working").rglob("best_mlp.pt")))
    preferred = [p for p in paths if "qwen2_5_vl_3b_shared" in str(p)]
    if len(preferred) == 1:
        return preferred[0]
    if len(paths) == 1:
        return paths[0]
    raise FileNotFoundError(
        "Set CHECKPOINT to the full-video+utterance MLP used earlier. "
        f"Found candidates: {[str(p) for p in paths]}"
    )


def rgb_frame(frame, width, height):
    a = frame.detach().cpu().numpy() if hasattr(frame, "detach") else np.asarray(frame)
    if a.ndim != 3:
        raise ValueError(f"Expected a sampled RGB frame, got {a.shape}")
    if a.shape[0] in (1, 3) and a.shape[-1] not in (1, 3):
        a = np.moveaxis(a, 0, -1)
    if a.dtype != np.uint8:
        if np.issubdtype(a.dtype, np.floating) and a.max() <= 1:
            a = a * 255
        a = np.clip(a, 0, 255).astype(np.uint8)
    return cv2.resize(a, (width, height), interpolation=cv2.INTER_LINEAR)


def face_ordinals(frames, grid, patch, merge, temporal_patch):
    """Union boxes from both frames of each tubelet and select overlapping cells."""
    t, h, w = grid
    if h % merge or w % merge:
        raise ValueError(f"Grid {grid} is incompatible with merge={merge}")
    rows, cols = h // merge, w // merge
    cell, width, height = patch * merge, w * patch, h * patch
    if len(frames) != t * temporal_patch:
        raise ValueError(f"Sampled {len(frames)} frames; grid needs {t * temporal_patch}")
    selected = set()
    boxes_by_frame = defaultdict(list)
    images = []
    for fi, frame in enumerate(frames):
        image = rgb_frame(frame, width, height)
        images.append(image)
        bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        detections = DeepFace.extract_faces(
            img_path=bgr, detector_backend=DETECTOR,
            enforce_detection=False, align=False,
        )
        pair = fi // temporal_patch
        for detection in detections:
            # DeepFace's no-face fallback is the whole frame at confidence zero.
            if float(detection.get("confidence", 0)) <= 0:
                continue
            a = detection["facial_area"]
            raw_x, raw_y, raw_w, raw_h = (int(a[k]) for k in ("x", "y", "w", "h"))
            x, y = max(raw_x, 0), max(raw_y, 0)
            bw = min(raw_x + raw_w, width) - x
            bh = min(raw_y + raw_h, height) - y
            if bw <= 0 or bh <= 0:
                continue
            boxes_by_frame[fi].append((x, y, bw, bh))
            for r in range(rows):
                for c in range(cols):
                    x0, y0 = c * cell, r * cell
                    area = max(0, min(x + bw, x0 + cell) - max(x, x0)) * max(
                        0, min(y + bh, y0 + cell) - max(y, y0)
                    )
                    if area >= OVERLAP_FRACTION * cell * cell:
                        selected.add(pair * rows * cols + r * cols + c)
    return sorted(selected), boxes_by_frame, images, rows, cols, cell


def matched_controls(face, t, per_pair, total, seed):
    rng = random.Random(seed)
    face_set = set(face)
    random_keep, nonface_keep = [], []
    for pair in range(t):
        start = pair * per_pair
        k = sum(start <= x < start + per_pair for x in face)
        universe = list(range(start, start + per_pair))
        others = [x for x in universe if x not in face_set]
        if k > len(others):
            raise ValueError(f"Pair {pair}: {k} face tokens but only {len(others)} nonface controls")
        random_keep.extend(rng.sample(universe, k))
        nonface_keep.extend(rng.sample(others, k))
    assert len(random_keep) == len(nonface_keep) == len(face)
    assert all(0 <= x < total for x in random_keep + nonface_keep)
    return sorted(random_keep), sorted(nonface_keep)


def draw_map(images, boxes, face, grid, rows, cols, cell, output):
    t = grid[0]
    per_pair = rows * cols
    fig, axes = plt.subplots(2, (t + 1) // 2, figsize=(17, 8))
    axes = np.asarray(axes).ravel()
    for pair in range(t):
        fi = pair * int(model.config.vision_config.temporal_patch_size)
        image = images[fi].copy()
        for jj in range(fi, fi + int(model.config.vision_config.temporal_patch_size)):
            for x, y, w, h in boxes[jj]:
                cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2)
        chosen = [x for x in face if x // per_pair == pair]
        for ordinal in chosen:
            r, c = divmod(ordinal % per_pair, cols)
            x0, y0 = c * cell, r * cell
            cv2.rectangle(image, (x0, y0), (x0 + cell - 1, y0 + cell - 1), (0, 255, 0), 2)
            cv2.putText(image, str(ordinal), (x0 + 1, y0 + 11),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.32, (0, 255, 0), 1)
        axes[pair].imshow(image)
        axes[pair].set_title(f"Pair {pair}: {len(chosen)} retained face tokens")
        axes[pair].axis("off")
    for ax in axes[t:]:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(output, dpi=140)
    plt.show()
    plt.close(fig)


def forward_condition(inputs, video_mask, selected, condition):
    """Prune at LM entry after vision encoding; keep original multimodal RoPE IDs."""
    n = int(video_mask.sum().item())
    if selected is not None and any(x < 0 or x >= n for x in selected):
        raise ValueError("Selected visual ordinal outside this video's range")
    state = {"calls": 0, "mask": inputs["attention_mask"], "selected": selected}
    if condition == "full":
        hook = None
    else:
        def prune(_module, args, kwargs):
            state["calls"] += 1
            emb = kwargs.get("inputs_embeds")
            if not isinstance(emb, torch.Tensor) or emb.ndim != 3 or emb.shape[0] != 1:
                raise ValueError("Expected one LM inputs_embeds tensor")
            visual = video_mask[0].nonzero().flatten().to(emb.device)
            if len(visual) != emb.shape[1] - int((~video_mask).sum()):
                raise ValueError("LM sequence and video placeholder mask disagree")
            chosen = state["selected"]
            if condition == "divprune_matched":
                chosen = select_diverse_token_indices(
                    emb[0, visual], retain_ratio=state["target_count"] / n,
                ).tolist()
                if len(chosen) != state["target_count"]:
                    raise ValueError("DivPrune selected the wrong token count")
                state["selected"] = chosen
            keep = (~video_mask[0]).to(emb.device).clone()
            if chosen:
                keep[visual[torch.as_tensor(chosen, device=emb.device)]] = True
            idx = keep.nonzero().flatten()
            kw = dict(kwargs)
            kw["inputs_embeds"] = emb.index_select(1, idx)
            for key in ("attention_mask", "position_ids", "cache_position"):
                value = kw.get(key)
                if isinstance(value, torch.Tensor) and value.shape[-1] == emb.shape[1]:
                    kw[key] = value.index_select(-1, idx.to(value.device))
                elif key in ("attention_mask", "position_ids"):
                    raise ValueError(f"Expected original-length {key} at LM entry")
            state["mask"] = kw["attention_mask"]
            return args, kw
        if condition == "divprune_matched":
            state["target_count"] = len(selected)
        hook = model.model.language_model.register_forward_pre_hook(prune, with_kwargs=True)
    try:
        with torch.inference_mode():
            out = model.model(**inputs, use_cache=False, return_dict=True)
        if hook is not None and state["calls"] != 1:
            raise RuntimeError(f"LM pruning hook ran {state['calls']} times")
        hidden = out.last_hidden_state
        embedding = pool_hidden_states(
            hidden.detach().float().cpu(), state["mask"].detach().cpu(), POOLING,
        ).unsqueeze(0)
        x = standardize_with_checkpoint(embedding, checkpoint)
        with torch.inference_mode():
            probs = mlp(x).softmax(dim=-1)[0].cpu().numpy()
        return probs, state["selected"]
    finally:
        if hook is not None:
            hook.remove()


if not INDEX_CSV.is_file():
    raise FileNotFoundError(f"Attach the benchmark-code dataset: {INDEX_CSV}")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
os.environ["VIDEO_MAX_PIXELS"] = str(FRAME_SIZE**2 * MAX_FRAMES)
checkpoint_path = choose_checkpoint()
checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
(OUTPUT_DIR / "run_config.json").write_text(json.dumps({
    "model_id": MODEL_ID, "checkpoint": str(checkpoint_path),
    "index_csv": str(INDEX_CSV), "start": START, "limit": LIMIT,
    "fps": FPS, "min_frames": MIN_FRAMES, "max_frames": MAX_FRAMES,
    "frame_size": FRAME_SIZE, "prompt_style": PROMPT_STYLE,
    "pooling": POOLING, "detector": DETECTOR,
    "deepface_backend_engine": os.environ["DEEPFACE_BACKEND_ENGINE"],
    "overlap_fraction": OVERLAP_FRACTION, "control_seeds": CONTROL_SEEDS,
}, indent=2))
labels = list(checkpoint["label_names"])
mlp = EmotionMLP(checkpoint["input_dim"], checkpoint["hidden_dim"], len(labels), checkpoint.get("dropout", 0))
mlp.load_state_dict(checkpoint["model_state_dict"])
mlp.eval()
dtype = torch.float16 if torch.cuda.is_available() else torch.float32
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    MODEL_ID, torch_dtype=dtype, device_map="auto" if torch.cuda.is_available() else "cpu",
).eval()
model.requires_grad_(False)
processor = AutoProcessor.from_pretrained(
    MODEL_ID, min_pixels=128 * 28 * 28, max_pixels=FRAME_SIZE**2,
)
print("Checkpoint:", checkpoint_path)
print("Labels:", labels, "| pooling:", POOLING)

with INDEX_CSV.open(newline="") as f:
    samples = list(csv.DictReader(f))[START:START + LIMIT]
if not samples:
    raise ValueError("No selected rows; adjust START/LIMIT")

records, diagnostics, skipped = [], [], []
for sample_i, row in enumerate(samples):
    sample_id = row["sample_id"]
    try:
        messages = make_message(
            row["video_path"], row["utterance"], FPS, MIN_FRAMES, MAX_FRAMES,
            FRAME_SIZE, PROMPT_STYLE, "video_text",
        )
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, videos, video_kwargs = process_vision_info(messages, return_video_kwargs=True)
        batch = processor(
            text=[text], images=image_inputs, videos=videos, padding=True,
            return_tensors="pt", **fix_video_kwargs(video_kwargs),
        )
        if len(videos) != 1 or batch["video_grid_thw"].shape[0] != 1:
            raise ValueError("Expected exactly one decoded video")
        grid = tuple(int(x) for x in batch["video_grid_thw"][0].tolist())
        vision = model.config.vision_config
        face, boxes, images, rows, cols, cell = face_ordinals(
            videos[0], grid, int(vision.patch_size),
            int(vision.spatial_merge_size), int(vision.temporal_patch_size),
        )
        total = grid[0] * rows * cols
        mask = batch["input_ids"].eq(model.config.video_token_id)
        if int(mask.sum()) != total:
            raise ValueError("Video grid and processor placeholder count disagree")
        if not face:
            skipped.append({"sample_id": sample_id, "reason": "No detected face tokens"})
            print(sample_id, "skipped: no detected face tokens")
            continue
        if SHOW_FIRST_MAP and sample_i == 0:
            draw_map(images, boxes, face, grid, rows, cols, cell, OUTPUT_DIR / "face_token_map.png")
        batch = batch.to(model.device)
        mask = mask.to(model.device)
        conditions = {"full": None, "face_only": face,
                      "no_video": [], "divprune_matched": face}
        for seed in CONTROL_SEEDS:
            ran, non = matched_controls(face, grid[0], rows * cols, total, seed)
            conditions[f"random_{seed}"] = ran
            conditions[f"nonface_{seed}"] = non
        sample_records = []
        for condition, selected in conditions.items():
            probabilities, actual_selected = forward_condition(batch, mask, selected, condition)
            true_label = row.get("emotion", "")
            result = {
                "sample_id": sample_id, "true_label": true_label,
                "dialogue_id": row.get("dialogue_id", ""),
                "condition": condition, "grid": str(grid),
                "visual_tokens_total": total,
                "tokens_retained": total if condition == "full" else len(actual_selected),
                "face_tokens_available": len(face),
                "predicted_label": labels[int(probabilities.argmax())],
                "correct": labels[int(probabilities.argmax())] == true_label,
                "confidence": float(probabilities.max()),
                "p_true": float(probabilities[labels.index(true_label)]) if true_label in labels else np.nan,
                **{f"p_{name}": float(probabilities[j]) for j, name in enumerate(labels)},
            }
            sample_records.append(result)
        original = next(r for r in sample_records if r["condition"] == "full")
        for result in sample_records:
            result["delta_p_true_vs_full"] = result["p_true"] - original["p_true"]
            result["delta_p_full_pred_vs_full"] = (
                result[f"p_{original['predicted_label']}"] - original["confidence"]
            )
            result["flipped_vs_full"] = result["predicted_label"] != original["predicted_label"]
        records.extend(sample_records)
        diagnostics.append({"sample_id": sample_id, "grid": grid, "face_ordinals": face,
                            "face_count_by_pair": [sum(i * rows * cols <= x < (i + 1) * rows * cols for x in face)
                                                   for i in range(grid[0])],
                            "detected_boxes_by_sampled_frame": {str(k): v for k, v in boxes.items()}})
        pd.DataFrame(records).to_csv(OUTPUT_DIR / "predictions.csv", index=False)
        print(f"{sample_id}: grid={grid}, face={len(face)}/{total}, full={original['predicted_label']}")
        display(pd.DataFrame(sample_records)[["condition", "tokens_retained", "predicted_label",
                                              "p_true", "delta_p_true_vs_full", "flipped_vs_full"]])
    except Exception as exc:
        skipped.append({"sample_id": sample_id, "reason": repr(exc)})
        print(sample_id, "failed:", repr(exc))
    finally:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

if not records:
    raise RuntimeError(f"No completed samples. Skipped: {skipped}")
detail = pd.DataFrame(records)
detail.to_csv(OUTPUT_DIR / "predictions.csv", index=False)
(OUTPUT_DIR / "token_diagnostics.json").write_text(json.dumps(diagnostics, indent=2))
pd.DataFrame(skipped, columns=["sample_id", "reason"]).to_csv(OUTPUT_DIR / "skipped.csv", index=False)

# Paired summary; meaningful metrics require more than a single example.
summary = detail.groupby("condition", as_index=False).agg(
    samples=("sample_id", "size"),
    mean_retained=("tokens_retained", "mean"),
    accuracy=("correct", "mean"),
    mean_p_true=("p_true", "mean"),
    mean_delta_p_true=("delta_p_true_vs_full", "mean"),
    flip_rate=("flipped_vs_full", "mean"),
)
if detail["sample_id"].nunique() >= 2:
    from sklearn.metrics import f1_score
    for i, item in summary.iterrows():
        matched = detail[detail["condition"] == item["condition"]]
        summary.loc[i, "macro_f1"] = f1_score(
            matched["true_label"], matched["predicted_label"],
            labels=labels, average="macro", zero_division=0,
        )
summary.to_csv(OUTPUT_DIR / "summary.csv", index=False)
display(summary.sort_values("condition"))

wide = detail.pivot(index="sample_id", columns="condition", values="p_true")
if len(wide) > 1:
    sample_dialogue = detail.drop_duplicates("sample_id").set_index("sample_id")["dialogue_id"]
    control_groups = {
        "random_mean": [f"random_{s}" for s in CONTROL_SEEDS],
        "nonface_mean": [f"nonface_{s}" for s in CONTROL_SEEDS],
        "divprune_matched": ["divprune_matched"],
        "full": ["full"],
        "no_video": ["no_video"],
    }
    comparisons = []
    rng = np.random.default_rng(2026)
    for name, columns in control_groups.items():
        difference = (wide["face_only"] - wide[columns].mean(axis=1)).dropna()
        dialogue = sample_dialogue.loc[difference.index]
        cluster_values = [difference[dialogue == key].to_numpy()
                          for key in dialogue.unique()]
        boot = np.array([
            np.concatenate([cluster_values[i] for i in rng.integers(0, len(cluster_values), len(cluster_values))]).mean()
            for _ in range(2000)
        ])
        comparisons.append({
            "face_only_minus": name,
            "samples": len(difference),
            "dialogues": len(cluster_values),
            "mean_delta_p_true": float(difference.mean()),
            "bootstrap_95_low": float(np.quantile(boot, 0.025)),
            "bootstrap_95_high": float(np.quantile(boot, 0.975)),
        })
    paired = pd.DataFrame(comparisons)
    paired.to_csv(OUTPUT_DIR / "paired_comparisons.csv", index=False)
    display(paired)
print("Saved:", OUTPUT_DIR)
