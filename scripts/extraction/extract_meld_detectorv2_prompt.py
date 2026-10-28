#!/usr/bin/env python3
"""Frozen Qwen2.5-VL-3B MELD embeddings with sampled-frame Detectorv2 cues."""

from __future__ import annotations

import argparse
import gc
import json
import os
import traceback
from pathlib import Path

# Support direct execution from a read-only Kaggle benchmark-code dataset.
import sys as _sys
if __package__ in (None, ""):
    _sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

os.environ.setdefault("FORCE_QWENVL_VIDEO_READER", "decord")

import numpy as np
import pandas as pd
import torch
from feat import Detectorv2
from qwen_vl_utils import process_vision_info
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

from scripts.extraction.extract_qwen_shared_embeddings import (
    LABELS, fix_video_kwargs, make_message, make_prompt, pool_hidden_states,
)

MODEL_ID = "Qwen/Qwen2.5-VL-3B-Instruct"
MOUTH = ("jawOpen", "mouthFunnel", "mouthPucker", "mouthSmileLeft", "mouthSmileRight")
AUS = ("AU01", "AU04", "AU06", "AU12", "AU15", "AU25", "AU26")
EMOTIONS = ("Anger", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--index-csv", type=Path, required=True)
    p.add_argument("--output-pt", type=Path, required=True)
    p.add_argument("--model-id", default=MODEL_ID)
    p.add_argument("--fps", type=float, default=6.0)
    p.add_argument("--min-frames", type=int, default=4)
    p.add_argument("--max-frames", type=int, default=64)
    p.add_argument("--frame-size", type=int, default=224)
    p.add_argument("--face-threshold", type=float, default=0.7)
    p.add_argument("--start", type=int, default=0)
    p.add_argument("--limit", type=int)
    p.add_argument("--save-every", type=int, default=25)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--retry-errors", action="store_true",
                   help="When resuming, retry previously failed clips.")
    return p.parse_args()


def iou(a, b):
    x0, y0 = max(a[0], b[0]), max(a[1], b[1])
    x1, y1 = min(a[2], b[2]), min(a[3], b[3])
    intersection = max(0, x1 - x0) * max(0, y1 - y0)
    aa, bb = (a[2]-a[0])*(a[3]-a[1]), (b[2]-b[0])*(b[3]-b[1])
    return intersection / max(aa + bb - intersection, 1e-9)


def face_tracks(fex):
    """Link faces by adjacent sampled-frame boxes; never use emotion to select speaker."""
    tracks = []
    for frame, group in fex.groupby("frame", sort=True):
        for _, row in group.iterrows():
            if not np.isfinite(row.get("FaceScore", np.nan)):
                continue
            box = (float(row.FaceRectX), float(row.FaceRectY),
                   float(row.FaceRectX + row.FaceRectWidth),
                   float(row.FaceRectY + row.FaceRectHeight))
            candidates = [(iou(box, t["box"]), n) for n, t in enumerate(tracks)
                          if int(frame) - t["frame"] <= 2 and t["frame"] != int(frame)]
            score, idx = max(candidates, default=(0, -1))
            if score < 0.25:
                tracks.append({"box": box, "frame": int(frame), "rows": [row]})
            else:
                tracks[idx]["box"] = box
                tracks[idx]["frame"] = int(frame)
                tracks[idx]["rows"].append(row)
    return tracks


def speaker_summary(fex, frame_count):
    tracks = face_tracks(fex)
    ranked = []
    for track in tracks:
        rows = sorted(track["rows"], key=lambda r: r["frame"])
        if not rows:
            continue
        # Jaw/mouth motion is only a proxy for speaking, never proof of identity.
        motion = []
        for col in ("jawOpen", "AU25", "AU26"):
            values = np.array([float(r.get(col, np.nan)) for r in rows])
            if len(values) > 1 and np.isfinite(values).all():
                motion.append(float(np.mean(np.abs(np.diff(values)))))
        coverage = len(rows) / frame_count
        area = float(np.median([r.FaceRectWidth * r.FaceRectHeight for r in rows]))
        score = coverage * (0.05 + sum(motion)) * np.sqrt(max(area, 1))
        ranked.append((score, rows, coverage, motion))
    ranked.sort(key=lambda item: item[0], reverse=True)
    if not ranked:
        return {"status": "no_face", "frames": []}
    top = ranked[0]
    # Do not assert speaker identity when tracks are short or similarly plausible.
    confident = top[2] >= 0.25 and (len(ranked) == 1 or top[0] >= 1.3 * ranked[1][0])
    if not confident:
        return {"status": "ambiguous_speaker", "frames": []}
    rows = top[1]
    frame_data = []
    for row in rows:
        values = {col: round(float(row[col]), 3) for col in MOUTH + AUS
                  if col in row and np.isfinite(row[col])}
        frame_data.append({"frame": int(row["frame"]), "values": values})
    def stat(col):
        vals = [float(r[col]) for r in rows if col in r and np.isfinite(r[col])]
        if not vals:
            return None
        return {"mean": round(float(np.mean(vals)), 3),
                "range": round(float(max(vals)-min(vals)), 3)}
    return {"status": "likely_speaker", "observed_frames": len(rows),
            "sampled_frames": frame_count, "selection": "face continuity + mouth/jaw motion",
            "features": {col: stat(col) for col in MOUTH + AUS if stat(col) is not None},
            "frames": frame_data}


def face_prompt(summary):
    guide = (
        "Detectorv2 guide: AU01 inner brow raise, AU04 brow lower, AU06 cheek raise, "
        "AU12 lip corner pull, AU15 lip corner depress, AU25 lips part, AU26 jaw drop; "
        "AU values are predicted probabilities in [0,1]. jawOpen, mouthFunnel, "
        "mouthPucker and mouthSmileLeft/Right are blendshape coefficients in [0,1] "
        "(0 little activation, 1 high activation). Frame indices refer to the sampled "
        "video frames in order. Range means observed max minus min, not a package threshold. "
        "Mouth movement may indicate speaking but does not prove speaker identity. "
        "Treat all measurements as uncertain cues, not emotion labels."
    )
    if summary["status"] != "likely_speaker":
        return guide + "\nSpeaker face measurements unavailable: " + summary["status"] + "."
    compact = {key: value for key, value in summary.items() if key != "frames"}
    compact["mouth_timeline"] = [
        {"frame": x["frame"], **{k: v for k, v in x["values"].items()
                                if k in ("jawOpen", "AU25", "AU26")}}
        for x in summary["frames"]
    ]
    return guide + "\nLikely speaker face: " + json.dumps(compact, separators=(",", ":"))


def save(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    torch.save(payload, tmp)
    tmp.replace(path)


def main():
    args = parse_args()
    if not 0 < args.face_threshold <= 1:
        raise ValueError("face-threshold must be in (0,1]")
    os.environ["VIDEO_MAX_PIXELS"] = str(args.frame_size**2 * args.max_frames)
    df = pd.read_csv(args.index_csv).iloc[args.start:]
    if args.limit is not None:
        df = df.head(args.limit)
    required = {"sample_id", "video_path", "utterance", "emotion", "emotion_id"}
    if not required.issubset(df.columns):
        raise ValueError(f"Missing index fields: {required - set(df.columns)}")
    config = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()
              if k not in {"resume", "retry_errors", "save_every"}}
    config["detector_package"] = "py-feat 2.1.5"
    config["detector_cudnn_enabled"] = False
    payload = {"embeddings": torch.empty((0, 0), dtype=torch.float16),
               "labels": torch.empty((0,), dtype=torch.long), "sample_ids": [],
               "metadata": [], "errors": [], "label_names": LABELS, "config": config}
    if args.resume and args.output_pt.exists():
        payload = torch.load(args.output_pt, map_location="cpu", weights_only=False)
        if payload.get("config") != config:
            raise ValueError("Resume configuration differs from saved output. Save the old .pt file "
                             "elsewhere and start a fresh extraction for this detector setting.")
        if args.retry_errors:
            payload["errors"] = []
    elif args.output_pt.exists():
        raise FileExistsError("Output exists; use --resume or choose another path")
    done = set(payload["sample_ids"]) | {e["sample_id"] for e in payload["errors"]}
    processor = AutoProcessor.from_pretrained(args.model_id, use_fast=False)
    qwen = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model_id, dtype=torch.float16, device_map="auto").eval()
    detector = Detectorv2(device="cuda" if torch.cuda.is_available() else "cpu",
                          identity_model=None)
    embeddings = [x for x in payload["embeddings"]]
    labels = payload["labels"].tolist()
    def checkpoint():
        payload["embeddings"] = torch.stack(embeddings) if embeddings else torch.empty((0, 0), dtype=torch.float16)
        payload["labels"] = torch.tensor(labels, dtype=torch.long)
        save(args.output_pt, payload)
    consecutive_errors = 0
    for count, row in enumerate(df.itertuples(index=False), 1):
        if row.sample_id in done:
            continue
        stage = "index_validation"
        try:
            if row.emotion not in LABELS or LABELS[int(row.emotion_id)] != row.emotion:
                raise ValueError("Index emotion and emotion_id disagree")
            stage = "video_decode"
            messages = make_message(str(row.video_path), str(row.utterance), args.fps,
                                    args.min_frames, args.max_frames, args.frame_size,
                                    "emotion_task", "video_text")
            images, videos, video_kwargs = process_vision_info(messages, return_video_kwargs=True)
            if not videos or len(videos) != 1:
                raise ValueError("Expected one decoded video")
            frames = videos[0]
            if not isinstance(frames, torch.Tensor) or frames.ndim != 4 or frames.shape[1] != 3:
                raise ValueError("Expected sampled video frames as TCHW RGB tensor")
            detector_frames = frames
            if frames.is_floating_point() and float(frames.max()) <= 1.0:
                detector_frames = (frames * 255).clamp(0, 255)
            # Py-Feat's ConvNeXt depthwise convolution can fail to find a cuDNN
            # engine on some Kaggle GPU stacks. Keep cuDNN available to Qwen.
            stage = "detectorv2"
            previous_cudnn = torch.backends.cudnn.enabled
            try:
                torch.backends.cudnn.enabled = False
                fex = detector.detect(detector_frames, data_type="tensor", batch_size=8,
                                      face_detection_threshold=args.face_threshold,
                                      progress_bar=False)
            finally:
                torch.backends.cudnn.enabled = previous_cudnn
            summary = speaker_summary(fex, len(frames))
            prompt = make_prompt(str(row.utterance), "emotion_task", "video_text") + "\n" + face_prompt(summary)
            messages[0]["content"][-1]["text"] = prompt
            rendered = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            kwargs = {"text": [rendered], "videos": videos, "padding": True, "return_tensors": "pt"}
            if images:
                kwargs["images"] = images
            kwargs.update(fix_video_kwargs(video_kwargs))
            stage = "qwen_processor"
            inputs = processor(**kwargs).to(qwen.device)
            stage = "qwen_forward"
            with torch.inference_mode():
                output = qwen.model(**inputs, use_cache=False, return_dict=True)
            embedding = pool_hidden_states(output.last_hidden_state.detach().float().cpu(),
                                           inputs["attention_mask"].cpu(), "last").half()
            embeddings.append(embedding)
            labels.append(int(row.emotion_id))
            payload["sample_ids"].append(row.sample_id)
            payload["metadata"].append({"sample_id": row.sample_id, "speaker_face": summary,
                                        "prompt": prompt, "sampled_frames": len(frames),
                                        "detector_cudnn_enabled": False})
            consecutive_errors = 0
            print(f"{count}: {row.sample_id} {summary['status']} ({len(frames)} frames)", flush=True)
        except Exception as exc:
            consecutive_errors += 1
            detail = traceback.format_exc()
            payload["errors"].append({"sample_id": row.sample_id, "stage": stage,
                                      "error": f"{type(exc).__name__}: {exc}",
                                      "traceback": detail})
            print(f"{count}: {row.sample_id} ERROR at {stage}: {exc}", flush=True)
            if consecutive_errors == 1:
                print(detail, flush=True)
            if "unable to find an engine" in str(exc).lower() or consecutive_errors >= 10:
                checkpoint()
                raise RuntimeError(f"Stopped after a systematic failure at {stage}; "
                                   f"inspect {args.output_pt} errors and traceback") from exc
        if count % args.save_every == 0:
            checkpoint()
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    checkpoint()
    print(f"Saved {len(embeddings)} embeddings; {len(payload['errors'])} errors: {args.output_pt}")


if __name__ == "__main__":
    main()
