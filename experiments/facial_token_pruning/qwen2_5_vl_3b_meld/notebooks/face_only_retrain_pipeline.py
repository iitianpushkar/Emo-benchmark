"""Train a new face-token-only MLP and compare it with a full-video MLP.

This is embedded verbatim in the companion Kaggle notebook.
"""
from __future__ import annotations

import csv
import json
import os
import random
import subprocess
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
OUTPUT_DIR = Path("/kaggle/working/face_only_retrained")
BASELINE_CHECKPOINT = None  # Set exact Kaggle path if discovery is ambiguous.
MODEL_ID = "Qwen/Qwen2.5-VL-3B-Instruct"
FPS, MIN_FRAMES, MAX_FRAMES, FRAME_SIZE = 6.0, 4, 64, 224
PROMPT_STYLE, POOLING = "emotion_task", "last"
DETECTOR, OVERLAP_FRACTION = "retinaface", 0.10
SPLITS = ("train", "dev", "test")
# None means the full official split. Change limits for a smoke test only.
LIMITS = {"train": None, "dev": None, "test": None}
STARTS = {"train": 0, "dev": 0, "test": 0}
SHOW_FIRST_MAP = True
SEED = 42

sys.path.insert(0, str(REPO))
from scripts.classification.evaluate_mlp import (
    EmotionMLP, compute_metrics, standardize_with_checkpoint,
)
from scripts.extraction.extract_qwen_shared_embeddings import (
    fix_video_kwargs, make_message, pool_hidden_states,
)


def find_baseline_checkpoint():
    if BASELINE_CHECKPOINT is not None:
        path = Path(BASELINE_CHECKPOINT)
        if not path.is_file():
            raise FileNotFoundError(path)
        return path
    paths = list(Path("/kaggle/input").rglob("best_mlp.pt")) + list(
        Path("/kaggle/working").rglob("best_mlp.pt")
    )
    preferred = [p for p in paths if "qwen2_5_vl_3b_shared" in str(p)]
    if len(preferred) == 1:
        return preferred[0]
    raise FileNotFoundError(
        "Set BASELINE_CHECKPOINT to the original full-video+utterance MLP. "
        f"Candidates: {[str(p) for p in paths]}"
    )


def rgb_frame(frame, width, height):
    a = frame.detach().cpu().numpy() if hasattr(frame, "detach") else np.asarray(frame)
    if a.ndim != 3:
        raise ValueError(f"Expected an RGB sampled frame, got {a.shape}")
    if a.shape[0] in (1, 3) and a.shape[-1] not in (1, 3):
        a = np.moveaxis(a, 0, -1)
    if a.dtype != np.uint8:
        if np.issubdtype(a.dtype, np.floating) and a.max() <= 1:
            a = a * 255
        a = np.clip(a, 0, 255).astype(np.uint8)
    return cv2.resize(a, (width, height), interpolation=cv2.INTER_LINEAR)


def face_ordinals(frames, grid, patch, merge, temporal_patch):
    t, h, w = grid
    if h % merge or w % merge or len(frames) != t * temporal_patch:
        raise ValueError("Sampled frames and Qwen grid disagree")
    rows, cols = h // merge, w // merge
    cell, width, height = patch * merge, w * patch, h * patch
    selected, boxes_by_frame, images = set(), defaultdict(list), []
    for fi, frame in enumerate(frames):
        image = rgb_frame(frame, width, height)
        images.append(image)
        detections = DeepFace.extract_faces(
            img_path=cv2.cvtColor(image, cv2.COLOR_RGB2BGR),
            detector_backend=DETECTOR, enforce_detection=False, align=False,
        )
        pair = fi // temporal_patch
        for detection in detections:
            # The DeepFace no-detection fallback has confidence zero.
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


def show_map(images, boxes, selected, grid, rows, cols, cell, temporal_patch, output):
    t = grid[0]
    per_pair = rows * cols
    fig, axes = plt.subplots(2, (t + 1) // 2, figsize=(17, 8))
    axes = np.asarray(axes).ravel()
    for pair in range(t):
        fi = pair * temporal_patch
        image = images[fi].copy()
        for frame_i in range(fi, fi + temporal_patch):
            for x, y, w, h in boxes[frame_i]:
                cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2)
        chosen = [x for x in selected if x // per_pair == pair]
        for ordinal in chosen:
            r, c = divmod(ordinal % per_pair, cols)
            x0, y0 = c * cell, r * cell
            cv2.rectangle(image, (x0, y0), (x0 + cell - 1, y0 + cell - 1), (0, 255, 0), 2)
            cv2.putText(image, str(ordinal), (x0 + 1, y0 + 11),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.32, (0, 255, 0), 1)
        axes[pair].imshow(image)
        axes[pair].set_title(f"Pair {pair}: {len(chosen)} face tokens")
        axes[pair].axis("off")
    for ax in axes[t:]:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(output, dpi=140)
    plt.show()
    plt.close(fig)


def extract_embedding(inputs, visual_mask, selected=None):
    """Use normal vision encoding; physically prune projected tokens at LM entry."""
    effective_mask = inputs["attention_mask"]
    calls = 0
    handle = None
    if selected is not None:
        visual = visual_mask[0].nonzero().flatten()
        if any(x < 0 or x >= len(visual) for x in selected):
            raise ValueError("Face ordinal outside the processor video grid")

        def hook(_module, args, kwargs):
            nonlocal effective_mask, calls
            calls += 1
            emb = kwargs.get("inputs_embeds")
            if not isinstance(emb, torch.Tensor) or emb.ndim != 3 or emb.shape[0] != 1:
                raise ValueError("Expected batch-one LM inputs_embeds")
            if emb.shape[1] != visual_mask.shape[1]:
                raise ValueError("LM embeddings and processor video mask have different lengths")
            keep = (~visual_mask[0]).to(emb.device).clone()
            if selected:
                ids = torch.as_tensor(selected, device=emb.device, dtype=torch.long)
                keep[visual.to(emb.device)[ids]] = True
            idx = keep.nonzero().flatten()
            kw = dict(kwargs)
            kw["inputs_embeds"] = emb.index_select(1, idx)
            for key in ("attention_mask", "position_ids", "cache_position"):
                value = kw.get(key)
                if isinstance(value, torch.Tensor) and value.shape[-1] == emb.shape[1]:
                    kw[key] = value.index_select(-1, idx.to(value.device))
                elif key in ("attention_mask", "position_ids"):
                    raise ValueError(f"Original-length {key} is required")
            effective_mask = kw["attention_mask"]
            return args, kw

        handle = model.model.language_model.register_forward_pre_hook(hook, with_kwargs=True)
    try:
        with torch.inference_mode():
            out = model.model(**inputs, use_cache=False, return_dict=True)
        if handle is not None and calls != 1:
            raise RuntimeError(f"Pruning hook ran {calls} times")
        hidden = out.last_hidden_state.detach().float().cpu()
        pooled = pool_hidden_states(hidden, effective_mask.detach().cpu(), POOLING)
        return pooled
    finally:
        if handle is not None:
            handle.remove()


def extract_sample(row, split, draw=False):
    messages = make_message(
        row["video_path"], row["utterance"], FPS, MIN_FRAMES, MAX_FRAMES,
        FRAME_SIZE, PROMPT_STYLE, "video_text",
    )
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    images_in, videos, video_kwargs = process_vision_info(messages, return_video_kwargs=True)
    batch = processor(
        text=[text], images=images_in, videos=videos, padding=True,
        return_tensors="pt", **fix_video_kwargs(video_kwargs),
    )
    if len(videos) != 1 or batch["video_grid_thw"].shape[0] != 1:
        raise ValueError("Expected one video")
    grid = tuple(int(x) for x in batch["video_grid_thw"][0].tolist())
    vision = model.config.vision_config
    temporal = int(vision.temporal_patch_size)
    face, boxes, images, rows, cols, cell = face_ordinals(
        videos[0], grid, int(vision.patch_size), int(vision.spatial_merge_size), temporal,
    )
    total = grid[0] * rows * cols
    visual_mask = batch["input_ids"].eq(model.config.video_token_id)
    if int(visual_mask.sum()) != total:
        raise ValueError("Qwen grid and video placeholder count disagree")
    if draw:
        show_map(images, boxes, face, grid, rows, cols, cell, temporal,
                 OUTPUT_DIR / "example_face_token_map.png")
    batch = batch.to(model.device)
    visual_mask = visual_mask.to(model.device)
    result = {
        "sample_id": row["sample_id"], "label": row["emotion"],
        "dialogue_id": row.get("dialogue_id", ""), "video_path": row["video_path"],
        "grid": grid, "visual_tokens_total": total,
        "face_tokens_retained": len(face), "face_ordinals": face,
        "face_embedding": extract_embedding(batch, visual_mask, face),
        "text_embedding": extract_embedding(batch, visual_mask, []),
    }
    if split != "train":
        result["full_embedding"] = extract_embedding(batch, visual_mask, None)
    return result


def make_payload(items, embedding_key, labels):
    if not items:
        raise RuntimeError("No extracted examples in this split")
    return {
        "embeddings": torch.stack([x[embedding_key] for x in items]),
        "labels": torch.tensor([labels.index(x["label"]) for x in items], dtype=torch.long),
        "sample_ids": [x["sample_id"] for x in items],
        "metadata": [{k: x[k] for k in ("dialogue_id", "video_path", "visual_tokens_total",
                                         "face_tokens_retained", "face_ordinals")}
                     for x in items],
        "label_names": labels,
    }


def predict_checkpoint(checkpoint, payload, labels):
    classifier = EmotionMLP(
        checkpoint["input_dim"], checkpoint["hidden_dim"], len(labels),
        checkpoint.get("dropout", 0),
    )
    classifier.load_state_dict(checkpoint["model_state_dict"])
    classifier.eval()
    x = standardize_with_checkpoint(payload["embeddings"].float(), checkpoint)
    with torch.inference_mode():
        probabilities = classifier(x).softmax(-1).cpu().numpy()
    return probabilities


if not REPO.is_dir():
    raise FileNotFoundError(f"Attach benchmark-code dataset: {REPO}")
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
baseline_path = find_baseline_checkpoint()
baseline_ckpt = torch.load(baseline_path, map_location="cpu", weights_only=False)
labels = list(baseline_ckpt["label_names"])
config = {
    "model_id": MODEL_ID, "baseline_checkpoint": str(baseline_path),
    "fps": FPS, "min_frames": MIN_FRAMES, "max_frames": MAX_FRAMES,
    "frame_size": FRAME_SIZE, "prompt_style": PROMPT_STYLE, "pooling": POOLING,
    "detector": DETECTOR, "overlap_fraction": OVERLAP_FRACTION,
    "starts": STARTS, "limits": LIMITS, "seed": SEED,
}
config_path = OUTPUT_DIR / "run_config.json"
if config_path.exists() and json.loads(config_path.read_text()) != config:
    raise ValueError("Existing output has different settings; use a new OUTPUT_DIR")
config_path.write_text(json.dumps(config, indent=2))

dtype = torch.float16 if torch.cuda.is_available() else torch.float32
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    MODEL_ID, torch_dtype=dtype,
    device_map="auto" if torch.cuda.is_available() else "cpu",
).eval()
model.requires_grad_(False)
processor = AutoProcessor.from_pretrained(
    MODEL_ID, min_pixels=128 * 28 * 28, max_pixels=FRAME_SIZE**2,
)
print("Baseline checkpoint:", baseline_path)
print("Labels:", labels)

payloads, failures = {}, []
first_map_drawn = False
for split in SPLITS:
    index_path = REPO / f"data/indexes/meld_{split}_index.csv"
    with index_path.open(newline="") as f:
        rows = list(csv.DictReader(f))
    start, limit = STARTS[split], LIMITS[split]
    rows = rows[start:] if limit is None else rows[start:start + limit]
    if not rows:
        raise ValueError(f"No selected {split} rows")
    split_dir = OUTPUT_DIR / "cache" / split
    split_dir.mkdir(parents=True, exist_ok=True)
    items = []
    for j, row in enumerate(rows, 1):
        cache = split_dir / f"{row['sample_id']}.pt"
        try:
            if cache.exists():
                item = torch.load(cache, map_location="cpu", weights_only=False)
                if item["sample_id"] != row["sample_id"] or item["label"] != row["emotion"]:
                    raise ValueError("Cached sample ID or label mismatch")
            else:
                draw = SHOW_FIRST_MAP and not first_map_drawn
                item = extract_sample(row, split, draw=draw)
                first_map_drawn |= draw
                temp = cache.with_suffix(".tmp")
                torch.save(item, temp)
                temp.replace(cache)
            items.append(item)
        except Exception as exc:
            failures.append({"split": split, "sample_id": row["sample_id"], "error": repr(exc)})
            print(split, row["sample_id"], "failed:", repr(exc))
        if j % 100 == 0:
            print(f"{split}: {j}/{len(rows)} visited; {len(items)} usable", flush=True)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    for representation in ("face", "text"):
        payloads[(split, representation)] = make_payload(
            items, f"{representation}_embedding", labels,
        )
        torch.save(payloads[(split, representation)],
                   OUTPUT_DIR / f"{representation}_{split}.pt")
    if split != "train":
        payloads[(split, "full")] = make_payload(items, "full_embedding", labels)
        torch.save(payloads[(split, "full")], OUTPUT_DIR / f"full_{split}.pt")
    print(f"{split}: {len(items)}/{len(rows)} usable; no-face clips kept as zero-visual-token inputs")

pd.DataFrame(failures, columns=["split", "sample_id", "error"]).to_csv(
    OUTPUT_DIR / "failures.csv", index=False,
)
del model
if torch.cuda.is_available():
    torch.cuda.empty_cache()

# Use the repository's existing train/dev/test MLP training protocol.
baseline_args = baseline_ckpt.get("args", {})
trained = {}
for representation in ("face", "text"):
    training_dir = OUTPUT_DIR / f"{representation}_mlp"
    command = [
        sys.executable, str(REPO / "scripts/classification/train_mlp.py"),
        "--train-pt", str(OUTPUT_DIR / f"{representation}_train.pt"),
        "--dev-pt", str(OUTPUT_DIR / f"{representation}_dev.pt"),
        "--test-pt", str(OUTPUT_DIR / f"{representation}_test.pt"),
        "--output-dir", str(training_dir),
        "--hidden-dim", str(baseline_args.get("hidden_dim", 512)),
        "--dropout", str(baseline_args.get("dropout", 0.3)),
        "--epochs", str(baseline_args.get("epochs", 50)),
        "--batch-size", str(baseline_args.get("batch_size", 64)),
        "--learning-rate", str(baseline_args.get("learning_rate", 3e-4)),
        "--weight-decay", str(baseline_args.get("weight_decay", 1e-4)),
        "--seed", str(SEED),
    ]
    if baseline_args.get("no_class_weights", False):
        command.append("--no-class-weights")
    print(f"Training {representation} MLP:", " ".join(command))
    subprocess.run(command, check=True)
    trained[representation] = torch.load(
        training_dir / "best_mlp.pt", map_location="cpu", weights_only=False,
    )
    if trained[representation]["label_names"] != labels:
        raise ValueError("Trained and baseline checkpoint label orders disagree")

metrics, prediction_rows, detailed_metrics = [], [], {}
for split in ("dev", "test"):
    face_payload = payloads[(split, "face")]
    text_payload = payloads[(split, "text")]
    full_payload = payloads[(split, "full")]
    if not (face_payload["sample_ids"] == text_payload["sample_ids"]
            == full_payload["sample_ids"]):
        raise ValueError("Face, text, and full examples are not paired")
    y = face_payload["labels"].numpy()
    for name, ckpt, payload in (
        ("full_video_saved_mlp", baseline_ckpt, full_payload),
        ("face_only_retrained_mlp", trained["face"], face_payload),
        ("utterance_only_retrained_mlp", trained["text"], text_payload),
    ):
        probabilities = predict_checkpoint(ckpt, payload, labels)
        pred = probabilities.argmax(axis=1)
        result = compute_metrics(y, pred, labels)
        detailed_metrics[f"{split}/{name}"] = result
        metrics.append({"split": split, "condition": name,
                        "samples": len(y), "accuracy": result["accuracy"],
                        "macro_f1": result["macro_f1"], "weighted_f1": result["weighted_f1"]})
        for i, sample_id in enumerate(payload["sample_ids"]):
            prediction_rows.append({
                "split": split, "sample_id": sample_id, "condition": name,
                "true_label": labels[int(y[i])], "predicted_label": labels[int(pred[i])],
                "correct": bool(y[i] == pred[i]),
                "p_true": float(probabilities[i, int(y[i])]),
                "dialogue_id": payload["metadata"][i]["dialogue_id"],
                "face_tokens_retained": payload["metadata"][i]["face_tokens_retained"],
                "visual_tokens_total": payload["metadata"][i]["visual_tokens_total"],
                **{f"p_{label}": float(probabilities[i, k]) for k, label in enumerate(labels)},
            })

comparison = pd.DataFrame(metrics)
comparison.to_csv(OUTPUT_DIR / "matched_metrics.csv", index=False)
pd.DataFrame(prediction_rows).to_csv(OUTPUT_DIR / "matched_predictions.csv", index=False)
(OUTPUT_DIR / "detailed_metrics.json").write_text(json.dumps(detailed_metrics, indent=2))
display(comparison)
print("Saved face-only MLP:", OUTPUT_DIR / "face_mlp/best_mlp.pt")
print("Saved utterance-only MLP:", OUTPUT_DIR / "text_mlp/best_mlp.pt")
print("Saved paired comparison:", OUTPUT_DIR / "matched_metrics.csv")
