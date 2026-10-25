#!/usr/bin/env python3
"""Train a MELD face-only MLP and compare it with the saved Qwen3B baseline.

Qwen is frozen. Face tokens are removed at the language-model entry, after
normal vision encoding. The face-only checkpoint is chosen on dev. The saved
full-video checkpoint and predictions are reused on paired dev/test samples.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib
import importlib.metadata
import json
import os
import random
import statistics
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
os.environ.setdefault("FORCE_QWENVL_VIDEO_READER", "decord")
os.environ.setdefault("DEEPFACE_BACKEND_ENGINE", "onnx")

from cluster.live_log import activate_live_log
from cluster.preflight import verify_meld
from cluster.verify_qwen_environment import require_cuda_gpu, verify_qwen_environment

MODEL_ID = "Qwen/Qwen2.5-VL-3B-Instruct"
LABELS = ["anger", "disgust", "fear", "joy", "neutral", "sadness", "surprise"]
SPLITS = ("train", "dev", "test")
CONTROL_SEEDS = (0, 1, 2, 3, 4)


def args_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=REPO.parents[1],
                        help="Parent of codes/, datasets/, models/, results/")
    parser.add_argument("--meld-root", type=Path, help="Default: ROOT/datasets/meld")
    parser.add_argument("--model-dir", type=Path, help="Default: ROOT/models/qwen2_5_vl_3b")
    parser.add_argument("--baseline-dir", type=Path,
                        help="Saved full-video MLP and dev/test predictions (default: repository baseline outputs)")
    parser.add_argument("--output-dir", type=Path,
                        help="Default: ROOT/results/qwen2_5_vl_3b_meld_face_only")
    parser.add_argument("--verify-only", action="store_true")
    parser.add_argument("--limit-per-split", type=int,
                        help="Smoke test: process at most N clips in each split")
    parser.add_argument("--skip-controls", action="store_true",
                        help="Skip the fixed-MLP dev ablation controls")
    parser.add_argument("--chunk-size", type=int, default=500,
                        help="Clips per resumable extraction chunk (default: 500)")
    parser.add_argument("--log-every", type=int, default=1)
    parser.add_argument("--profile-every", type=int, default=100,
                        help="Optional counted-operator TFLOPs profile every Nth clip (0 disables)")
    parser.add_argument("--epochs", type=int,
                        help="Face MLP epochs (default: saved baseline training setting)")
    parser.add_argument("--fps", type=float, default=6.0)
    parser.add_argument("--min-frames", type=int, default=4)
    parser.add_argument("--max-frames", type=int, default=64)
    parser.add_argument("--frame-size", type=int, default=224)
    parser.add_argument("--overlap-fraction", type=float, default=0.10)
    parser.add_argument("--seed", type=int, default=42)
    return parser


def verify_model_3b(root: Path) -> dict:
    config_path = root / "config.json"
    if not config_path.is_file():
        raise FileNotFoundError(f"Qwen config missing: {config_path}")
    config = json.loads(config_path.read_text())
    text_config = config.get("text_config", config)
    vision_config = config.get("vision_config", {})
    signature = (config.get("model_type"), text_config.get("hidden_size"),
                 text_config.get("num_hidden_layers"), vision_config.get("out_hidden_size"),
                 vision_config.get("patch_size"))
    if signature != ("qwen2_5_vl", 2048, 36, 2048, 14):
        raise ValueError(f"{root} is not Qwen2.5-VL-3B-Instruct: {signature}")
    for name in ("preprocessor_config.json", "tokenizer_config.json"):
        if not (root / name).is_file():
            raise FileNotFoundError(root / name)
    shards = sorted(root.glob("*.safetensors"))
    if not shards or any(p.stat().st_size < 1_000_000 for p in shards):
        raise ValueError(f"Missing, incomplete, or LFS-pointer weight shards at {root}")
    index = root / "model.safetensors.index.json"
    if index.is_file():
        data = json.loads(index.read_text())
        expected = set(data["weight_map"].values())
        if any(not (root / name).is_file() for name in expected):
            raise ValueError("Qwen model index references missing weight shards")
        if data.get("metadata", {}).get("total_size"):
            if sum((root / name).stat().st_size for name in expected) < data["metadata"]["total_size"]:
                raise ValueError("Qwen weight shards appear truncated")
    return {"path": str(root), "signature": signature, "shards": len(shards)}


def verify_face_dependencies() -> dict:
    prefix = Path(sys.prefix).resolve()
    found = {}
    print("Checking face-detector imports in the active virtual environment", flush=True)
    for package, module in (("Werkzeug", "werkzeug"), ("deepface", "deepface"),
                            ("pandas", "pandas")):
        try:
            dist = importlib.metadata.distribution(package)
            location = Path(dist.locate_file("")).resolve()
            if not location.is_relative_to(prefix):
                raise RuntimeError(f"{package} is outside virtual environment: {location}")
            imported = importlib.import_module(module)
            module_path = Path(imported.__file__).resolve()
            if not module_path.is_relative_to(prefix):
                raise RuntimeError(f"{module} is outside virtual environment: {module_path}")
            if package == "deepface":
                # DeepFace is a submodule; importing the package alone does not
                # necessarily expose it as an attribute until it is imported.
                api = importlib.import_module("deepface.DeepFace")
                api_path = Path(api.__file__).resolve()
                if not api_path.is_relative_to(prefix):
                    raise RuntimeError(f"deepface.DeepFace is outside virtual environment: {api_path}")
                for name in ("build_model", "extract_faces"):
                    if not callable(getattr(api, name, None)):
                        raise RuntimeError(f"deepface.DeepFace.{name} is unavailable")
            found[package] = dist.version
            print(f"  {package}: {dist.version} ({module_path})", flush=True)
        except Exception as exc:
            raise RuntimeError(f"Face detector dependency {package} failed: {exc}") from exc
    return found


def verify_retinaface_weights() -> Path:
    """Require the ONNX detector weights before entering an offline GPU job."""
    home = Path(os.environ.get("DEEPFACE_HOME", str(Path.home()))).expanduser()
    weights = home / ".deepface" / "weights" / "retinaface.onnx"
    if not weights.is_file() or weights.stat().st_size < 100_000_000:
        raise FileNotFoundError(
            f"RetinaFace ONNX weights missing or incomplete: {weights}. "
            "Download serengil/deepface retinaface.onnx on a networked login/transfer "
            "node into this directory, then run --verify-only again."
        )
    print(f"RetinaFace weights: {weights} ({weights.stat().st_size:,} bytes)", flush=True)
    return weights


def build_retinaface_for_allocation(deepface) -> dict:
    """Build DeepFace's ONNX detector using the CPUs assigned by Slurm."""
    import onnxruntime as ort

    available = len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else (os.cpu_count() or 1)
    requested = int(os.environ.get("SLURM_CPUS_PER_TASK", available))
    threads = max(1, min(available, requested))
    original_options = ort.SessionOptions

    def limited_options(*args, **kwargs):
        options = original_options(*args, **kwargs)
        options.intra_op_num_threads = threads
        options.inter_op_num_threads = 1
        return options

    # retinaface's ONNX loader constructs SessionOptions internally. Replace
    # that factory only while DeepFace builds its singleton detector, then
    # restore it so other ONNX users keep their own configuration.
    ort.SessionOptions = limited_options
    try:
        detector = deepface.build_model("retinaface", task="face_detector")
    finally:
        ort.SessionOptions = original_options
    session = getattr(getattr(detector, "model", None), "session", None)
    if session is None:
        raise RuntimeError("RetinaFace did not expose its ONNX session; thread limit is unverified")
    runtime = {"onnxruntime": ort.__version__, "intra_op_threads": threads,
               "inter_op_threads": 1, "providers": session.get_providers()}
    print(f"RetinaFace runtime: {runtime}", flush=True)
    return runtime


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_baseline_predictions(folder: Path) -> dict[str, dict[str, dict]]:
    predictions = {}
    required = {"sample_id", "true_label", "pred_label"} | {f"prob_{name}" for name in LABELS}
    for split in ("dev", "test"):
        path = folder / f"{split}_predictions.csv"
        if not path.is_file():
            raise FileNotFoundError(f"Saved full-video predictions missing: {path}")
        with path.open(newline="", encoding="utf-8") as file:
            reader = csv.DictReader(file)
            if not required.issubset(reader.fieldnames or []):
                raise ValueError(f"Saved predictions have incompatible columns: {path}")
            rows = list(reader)
        ids = [row["sample_id"] for row in rows]
        if len(ids) != len(set(ids)):
            raise ValueError(f"Duplicate baseline sample IDs: {path}")
        if any(row["true_label"] not in LABELS or row["pred_label"] not in LABELS for row in rows):
            raise ValueError(f"Incompatible baseline labels: {path}")
        predictions[split] = {row["sample_id"]: row for row in rows}
        print(f"Saved full-video {split}: {len(rows)} predictions at {path}", flush=True)
    return predictions


def code_revision() -> str | None:
    result = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO,
                            text=True, capture_output=True, check=False)
    return result.stdout.strip() if result.returncode == 0 else None


def validate_resume_config(previous: dict, current: dict, output: Path) -> None:
    """Permit known operational updates without mixing incompatible embeddings."""
    # run_config.json turns tuples (such as CONTROL_SEEDS) into lists. Compare
    # the serialized form so a resume does not mistake that for a setting change.
    current = json.loads(json.dumps(current))
    operational = {"code_revision", "retinaface_runtime"}
    previous_revision = str(previous.get("code_revision") or "")
    missing_weight_hash = "retinaface_weights_sha256" not in previous
    if missing_weight_hash and previous_revision.startswith("6b076c1"):
        # The first resumable version did not record the detector weight hash.
        # Its outputs cannot be tied to the staged weights, so reuse is safe
        # only if no embedding or classifier artifact has been written yet.
        artifacts = (list((output / "cache").rglob("*.pt"))
                     + list((output / "chunks").rglob("*.pt"))
                     + list(output.glob("face_*.pt")))
        if not artifacts:
            operational.add("retinaface_weights_sha256")
    old_science = {key: value for key, value in previous.items() if key not in operational}
    new_science = {key: value for key, value in current.items() if key not in operational}
    differences = {key: {"previous": old_science.get(key, "<missing>"),
                         "current": new_science.get(key, "<missing>")}
                   for key in sorted(old_science.keys() | new_science.keys())
                   if old_science.get(key, "<missing>") != new_science.get(key, "<missing>")}
    compatible_revision = (previous_revision.startswith("aefe030")
                           or previous_revision.startswith("6b076c1")
                           or previous_revision == str(current.get("code_revision")))
    if differences or not compatible_revision:
        if not compatible_revision:
            differences["code_revision"] = {"previous": previous.get("code_revision"),
                                            "current": current.get("code_revision")}
        raise ValueError(
            f"Output directory has different run settings: {output / 'run_config.json'}\n"
            f"Differences: {json.dumps(differences, indent=2, default=str)}\n"
            "Use a new --output-dir for a different experiment; do not delete cached embeddings."
        )
    print(f"Resuming {output} with compatible run settings", flush=True)


def run_live(command: list[str]) -> None:
    """Mirror child output through the Python tee as well as job stdout."""
    print("Running:", " ".join(command), flush=True)
    with subprocess.Popen(command, cwd=REPO, env={**os.environ, "PYTHONUNBUFFERED": "1",
                                                  "HF_HUB_OFFLINE": "1"},
                          stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                          text=True, bufsize=1) as process:
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="", flush=True)
        status = process.wait()
    if status:
        raise subprocess.CalledProcessError(status, command)


def atomic_json(path: Path, value: object) -> None:
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2) + "\n")
    temp.replace(path)


def atomic_torch(path: Path, value: object, torch) -> None:
    temp = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, temp)
    temp.replace(path)


def valid_chunk(payload, split: str, start: int, end: int, rows) -> bool:
    """Check chunk identity and coverage before reusing or merging it."""
    if not isinstance(payload, dict):
        return False
    expected = [row["sample_id"] for row in rows[start:end]]
    if (payload.get("split") != split or payload.get("start") != start
            or payload.get("end") != end or payload.get("expected_sample_ids") != expected):
        return False
    items = payload.get("items")
    failures = payload.get("failures")
    if not isinstance(items, list) or not isinstance(failures, list):
        return False
    try:
        done = [item["sample_id"] for item in items]
        failed = [item["sample_id"] for item in failures]
    except (KeyError, TypeError):
        return False
    if len(done) + len(failed) != len(expected) or set(done + failed) != set(expected):
        return False
    expected_labels = {row["sample_id"]: row["emotion"] for row in rows[start:end]}
    try:
        return all(item["label"] == expected_labels[item["sample_id"]]
                   and "face" in item["embeddings"] and "face" in item["costs"]
                   for item in items)
    except (KeyError, TypeError):
        return False


def face_ordinals(frames, grid, vision, deepface, cv2, np, overlap, progress_label=None):
    """Union faces from both frames of each temporal Qwen tubelet."""
    t, h, w = grid
    merge = int(vision.spatial_merge_size)
    temporal = int(vision.temporal_patch_size)
    if h % merge or w % merge or len(frames) != t * temporal:
        raise ValueError(f"Sampled frames ({len(frames)}) disagree with video grid {grid}")
    rows, cols = h // merge, w // merge
    cell = int(vision.patch_size) * merge
    width, height = w * int(vision.patch_size), h * int(vision.patch_size)
    selected, boxes = set(), defaultdict(list)
    detection_started = last_progress = time.perf_counter()
    for fi, frame in enumerate(frames):
        image = frame.detach().cpu().numpy() if hasattr(frame, "detach") else np.asarray(frame)
        if image.ndim != 3:
            raise ValueError(f"Frame has unexpected shape {image.shape}")
        if image.shape[0] in (1, 3) and image.shape[-1] not in (1, 3):
            image = np.moveaxis(image, 0, -1)
        if image.dtype != np.uint8:
            if np.issubdtype(image.dtype, np.floating) and image.max() <= 1:
                image = image * 255
            image = np.clip(image, 0, 255).astype(np.uint8)
        image = cv2.resize(image, (width, height), interpolation=cv2.INTER_LINEAR)
        detected = deepface.extract_faces(
            img_path=cv2.cvtColor(image, cv2.COLOR_RGB2BGR),
            detector_backend="retinaface", enforce_detection=False, align=False,
        )
        now = time.perf_counter()
        if progress_label and now - last_progress >= 30:
            print(f"{progress_label}: RetinaFace {fi + 1}/{len(frames)} frames "
                  f"in {now - detection_started:.1f}s", flush=True)
            last_progress = now
        for face in detected:
            if float(face.get("confidence", 0)) <= 0:
                continue
            area = face["facial_area"]
            raw_x, raw_y, raw_w, raw_h = (int(area[k]) for k in ("x", "y", "w", "h"))
            x, y = max(raw_x, 0), max(raw_y, 0)
            bw, bh = min(raw_x + raw_w, width) - x, min(raw_y + raw_h, height) - y
            if bw <= 0 or bh <= 0:
                continue
            boxes[fi].append((x, y, bw, bh))
            pair = fi // temporal
            for r in range(rows):
                for c in range(cols):
                    x0, y0 = c * cell, r * cell
                    intersect = max(0, min(x + bw, x0 + cell) - max(x, x0)) * max(
                        0, min(y + bh, y0 + cell) - max(y, y0))
                    if intersect >= overlap * cell * cell:
                        selected.add(pair * rows * cols + r * cols + c)
    return sorted(selected), {str(k): v for k, v in boxes.items()}, rows * cols


def matched_controls(face: list[int], t: int, per_pair: int, seed: int):
    rng = random.Random(seed)
    face_set = set(face)
    random_keep, nonface_keep = [], []
    nonface_possible = True
    for pair in range(t):
        start = pair * per_pair
        k = sum(start <= x < start + per_pair for x in face)
        universe = list(range(start, start + per_pair))
        other = [x for x in universe if x not in face_set]
        random_keep.extend(rng.sample(universe, k))
        if k > len(other):
            nonface_possible = False
        elif nonface_possible:
            nonface_keep.extend(rng.sample(other, k))
    return sorted(random_keep), sorted(nonface_keep) if nonface_possible else None


def forward(model, batch, mask, selected, torch, divprune=False, profile=False):
    """Run one frozen forward and prune projected tokens before the LM."""
    visual = mask[0].nonzero().flatten()
    n = int(visual.numel())
    if selected is not None and any(i < 0 or i >= n for i in selected):
        raise ValueError("Visual ordinal outside processor video grid")
    state = {"calls": 0, "effective_mask": batch["attention_mask"], "selected": selected}
    hook = None
    if selected is not None:
        def prune(_module, args, kwargs):
            state["calls"] += 1
            emb = kwargs.get("inputs_embeds")
            if not isinstance(emb, torch.Tensor) or emb.ndim != 3 or emb.shape[0] != 1:
                raise ValueError("Expected batch-one LM inputs_embeds")
            if emb.shape[1] != mask.shape[1]:
                raise ValueError("LM embeddings and video mask differ in length")
            chosen = selected
            if divprune:
                from scripts.pruning.divprune import select_diverse_token_indices
                chosen = select_diverse_token_indices(
                    emb[0, visual.to(emb.device)], retain_ratio=len(selected) / n,
                ).tolist()
                if len(chosen) != len(selected):
                    raise ValueError("DivPrune selected the wrong number of tokens")
                state["selected"] = chosen
            keep = (~mask[0]).to(emb.device).clone()
            if chosen:
                keep[visual.to(emb.device)[torch.tensor(chosen, device=emb.device)]] = True
            idx = keep.nonzero().flatten()
            kw = dict(kwargs)
            kw["inputs_embeds"] = emb.index_select(1, idx)
            for key in ("attention_mask", "position_ids", "cache_position"):
                value = kw.get(key)
                if isinstance(value, torch.Tensor) and value.shape[-1] == emb.shape[1]:
                    kw[key] = value.index_select(-1, idx.to(value.device))
                elif key in ("attention_mask", "position_ids"):
                    raise ValueError(f"Expected original-length {key} at LM entry")
            state["effective_mask"] = kw["attention_mask"]
            return args, kw
        hook = model.model.language_model.register_forward_pre_hook(prune, with_kwargs=True)
    try:
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
        started = time.perf_counter()
        with torch.inference_mode():
            if profile:
                from torch.utils.flop_counter import FlopCounterMode
                with FlopCounterMode(display=False) as counter:
                    output = model.model(**batch, use_cache=False, return_dict=True)
                flops = int(counter.get_total_flops())
            else:
                output = model.model(**batch, use_cache=False, return_dict=True)
                flops = None
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - started
        if hook is not None and state["calls"] != 1:
            raise RuntimeError(f"LM hook ran {state['calls']} times")
        from scripts.extraction.extract_qwen_shared_embeddings import pool_hidden_states
        pooled = pool_hidden_states(
            output.last_hidden_state.detach().float().cpu(),
            state["effective_mask"].detach().cpu(), "last",
        )
        cost = {"forward_seconds": elapsed, "profiler_active": profile,
                "full_visual_tokens": n,
                "retained_visual_tokens": n if selected is None else len(state["selected"]),
                "full_sequence_tokens": int(mask.shape[1]),
                "retained_sequence_tokens": int(state["effective_mask"].shape[1]),
                "peak_allocated_bytes": torch.cuda.max_memory_allocated() if torch.cuda.is_available() else None,
                "counted_operator_tflops": flops / 1e12 if flops is not None else None,
                "flops_scope": "vision encoder + LM forward; PyTorch registered operators only"}
        return pooled, cost
    finally:
        if hook is not None:
            hook.remove()


def extract_one(row, split, model, processor, args, torch, np, cv2, deepface, process_vision_info):
    from scripts.extraction.extract_qwen_shared_embeddings import fix_video_kwargs, make_message
    started = time.perf_counter()
    trace = args.current_sample == 1 or args.current_sample % 100 == 0
    progress_label = f"{split}/{row['sample_id']}"
    messages = make_message(row["video_path"], row["utterance"], args.fps,
                            args.min_frames, args.max_frames, args.frame_size,
                            "emotion_task", "video_text")
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    if trace:
        print(f"{progress_label}: decoding video", flush=True)
    images, videos, video_kwargs = process_vision_info(messages, return_video_kwargs=True)
    if trace:
        print(f"{progress_label}: decoded {len(videos[0]) if videos else 0} frames "
              f"in {time.perf_counter() - started:.1f}s; processing Qwen input", flush=True)
    batch = processor(text=[text], images=images, videos=videos, padding=True,
                      return_tensors="pt", **fix_video_kwargs(video_kwargs))
    if len(videos) != 1 or batch["video_grid_thw"].shape[0] != 1:
        raise ValueError("Expected one video and one Qwen video grid")
    processed = time.perf_counter()
    grid = tuple(int(x) for x in batch["video_grid_thw"][0].tolist())
    if trace:
        print(f"{progress_label}: Qwen input ready in {processed - started:.1f}s; "
              f"detecting faces on {len(videos[0])} frames", flush=True)
    face, boxes, per_pair = face_ordinals(
        videos[0], grid, model.config.vision_config, deepface, cv2, np,
        args.overlap_fraction, progress_label,
    )
    detected = time.perf_counter()
    if trace:
        print(f"{progress_label}: face detection finished in {detected - processed:.1f}s; "
              f"retained {len(face)} visual tokens", flush=True)
    mask = batch["input_ids"].eq(model.config.video_token_id)
    total = grid[0] * per_pair
    if int(mask.sum()) != total:
        raise ValueError("Qwen grid and processor video placeholders disagree")
    batch, mask = batch.to(model.device), mask.to(model.device)
    conditions = {"face": face}
    if split == "dev" and not args.skip_controls and face:
        conditions["text"] = []
        conditions["divprune_matched"] = face
        for seed in CONTROL_SEEDS:
            random_keep, nonface_keep = matched_controls(face, grid[0], per_pair, seed)
            conditions[f"random_{seed}"] = random_keep
            if nonface_keep is not None:
                conditions[f"nonface_{seed}"] = nonface_keep
    embeddings, costs = {}, {}
    for name, selected in conditions.items():
        if trace:
            print(f"{progress_label}: Qwen forward for {name}", flush=True)
        embeddings[name], costs[name] = forward(
            model, batch, mask, selected, torch,
            divprune=name == "divprune_matched",
            profile=args.profile_every > 0 and args.current_sample % args.profile_every == 0,
        )
    finished = time.perf_counter()
    if trace:
        print(f"{progress_label}: extraction finished in {finished - started:.1f}s", flush=True)
    return {"sample_id": row["sample_id"], "label": row["emotion"],
            "dialogue_id": row["dialogue_id"], "video_path": row["video_path"],
            "grid": grid, "face_ordinals": face, "boxes": boxes,
            "embeddings": embeddings, "costs": costs,
            "preprocess_seconds": processed - started,
            "face_detection_seconds": detected - processed,
            "clip_all_conditions_seconds": finished - started}


def make_payload(items, representation, torch):
    selected = [item for item in items if representation in item["embeddings"]]
    if not selected:
        raise RuntimeError(f"No {representation} embeddings extracted")
    return {"embeddings": torch.stack([x["embeddings"][representation] for x in selected]),
            "labels": torch.tensor([LABELS.index(x["label"]) for x in selected]),
            "label_names": LABELS,
            "sample_ids": [x["sample_id"] for x in selected],
            "metadata": [{"dialogue_id": x["dialogue_id"],
                          "video_file": x["video_path"],
                          "face_tokens_retained": len(x["face_ordinals"]),
                          "visual_tokens_total": x["costs"]["face"]["full_visual_tokens"],
                          "original_visual_tokens": x["costs"][representation]["full_visual_tokens"],
                          "retained_visual_tokens": x["costs"][representation]["retained_visual_tokens"]}
                         for x in selected]}


def train_face_mlp(output: Path, args, baseline_checkpoint, torch):
    folder = output / "face_mlp"
    checkpoint = folder / "best_mlp.pt"
    if checkpoint.is_file():
        print(f"Reusing {checkpoint}", flush=True)
        return torch.load(checkpoint, map_location="cpu", weights_only=False)
    baseline_args = baseline_checkpoint.get("args", {})
    command = [sys.executable, "-u", str(REPO / "scripts/classification/train_mlp.py"),
               "--train-pt", str(output / "face_train.pt"),
               "--dev-pt", str(output / "face_dev.pt"),
               "--test-pt", str(output / "face_test.pt"),
               "--output-dir", str(folder),
               "--hidden-dim", str(baseline_args.get("hidden_dim", 512)),
               "--dropout", str(baseline_args.get("dropout", 0.3)),
               "--epochs", str(args.epochs or baseline_args.get("epochs", 50)),
               "--batch-size", str(baseline_args.get("batch_size", 64)),
               "--learning-rate", str(baseline_args.get("learning_rate", 3e-4)),
               "--weight-decay", str(baseline_args.get("weight_decay", 1e-4)),
               "--seed", str(args.seed), "--device", "cuda"]
    if baseline_args.get("no_class_weights", False):
        command.append("--no-class-weights")
    run_live(command)
    return torch.load(checkpoint, map_location="cpu", weights_only=False)


def predict(checkpoint, payload, torch):
    from scripts.classification.evaluate_mlp import EmotionMLP, standardize_with_checkpoint
    classifier = EmotionMLP(checkpoint["input_dim"], checkpoint["hidden_dim"],
                            len(LABELS), checkpoint.get("dropout", 0))
    classifier.load_state_dict(checkpoint["model_state_dict"])
    classifier.eval()
    x = standardize_with_checkpoint(payload["embeddings"].float(), checkpoint)
    with torch.inference_mode():
        return classifier(x).softmax(-1).numpy()


def evaluate(output, payloads, baseline_checkpoint, face_checkpoint,
             baseline_predictions, args, torch, np):
    from scripts.classification.evaluate_mlp import compute_metrics
    rows, predictions = [], []

    def record(split, condition, payload, probabilities):
        y = payload["labels"].numpy()
        pred = probabilities.argmax(1)
        metrics = compute_metrics(y, pred, LABELS)
        original_counts = [int(meta["original_visual_tokens"]) for meta in payload["metadata"]]
        retained_counts = [int(meta["retained_visual_tokens"]) for meta in payload["metadata"]]
        if condition == "full_video_saved_mlp":
            # The historical full-video run did not record per-clip counts.
            # This unpruned count uses the current run's matching video input.
            retained_counts = original_counts.copy()
        if any(not 0 <= retained <= original for original, retained
               in zip(original_counts, retained_counts)):
            raise ValueError(f"Invalid visual-token counts for {split}/{condition}")
        rows.append({"split": split, "condition": condition, "samples": len(y),
                     "accuracy": metrics["accuracy"], "macro_f1": metrics["macro_f1"],
                     "weighted_f1": metrics["weighted_f1"],
                     "original_visual_tokens": sum(original_counts),
                     "retained_visual_tokens": sum(retained_counts),
                     "weighted_visual_token_retention": sum(retained_counts) / sum(original_counts)})
        for i, sample_id in enumerate(payload["sample_ids"]):
            predictions.append({"split": split, "sample_id": sample_id,
                                "condition": condition, "true_label": LABELS[y[i]],
                                "predicted_label": LABELS[pred[i]],
                                "p_true": float(probabilities[i, y[i]]),
                                "dialogue_id": payload["metadata"][i]["dialogue_id"],
                                "original_visual_tokens": original_counts[i],
                                "retained_visual_tokens": retained_counts[i],
                                "visual_token_retention": retained_counts[i] / original_counts[i]})

    for split in ("dev", "test"):
        payload = payloads[(split, "face")]
        saved_rows = [baseline_predictions[split][sample_id] for sample_id in payload["sample_ids"]]
        for i, row in enumerate(saved_rows):
            if row["true_label"] != LABELS[int(payload["labels"][i])]:
                raise ValueError(f"Baseline/face label mismatch: {split}/{row['sample_id']}")
        saved_probabilities = np.array([[float(row[f"prob_{label}"]) for label in LABELS]
                                        for row in saved_rows])
        if any(LABELS[int(saved_probabilities[i].argmax())] != row["pred_label"]
               for i, row in enumerate(saved_rows)):
            raise ValueError(f"Saved full-video probabilities disagree with predicted labels on {split}")
        record(split, "full_video_saved_mlp", payload, saved_probabilities)
        record(split, "face_only_retrained_mlp", payload,
               predict(face_checkpoint, payload, torch))
    if not args.skip_controls:
        for name in ["divprune_matched"] + [f"{kind}_{seed}" for kind in ("random", "nonface")
                                                for seed in CONTROL_SEEDS]:
            if ("dev", name) not in payloads:
                continue
            payload = payloads[("dev", name)]
            record("dev", name + "_fixed_mlp", payload,
                   predict(baseline_checkpoint, payload, torch))
        # No new utterance-only MLP is trained. This is the original
        # no-video input passed through the saved full-video classifier.
        for name in ("face", "text"):
            if ("dev", name) not in payloads:
                continue
            payload = payloads[("dev", name)]
            record("dev", name + "_fixed_mlp", payload,
                   predict(baseline_checkpoint, payload, torch))
    with (output / "matched_metrics.csv").open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    with (output / "matched_predictions.csv").open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(predictions[0]))
        writer.writeheader(); writer.writerows(predictions)
    atomic_json(output / "matched_metrics.json", rows)
    if not args.skip_controls:
        by_condition = defaultdict(dict)
        dialogues = {}
        for row in predictions:
            if row["split"] == "dev":
                by_condition[row["condition"]][row["sample_id"]] = row["p_true"]
                dialogues[row["sample_id"]] = row["dialogue_id"]
        groups = {"random_mean": [f"random_{seed}_fixed_mlp" for seed in CONTROL_SEEDS],
                  "nonface_mean": [f"nonface_{seed}_fixed_mlp" for seed in CONTROL_SEEDS],
                  "divprune_matched": ["divprune_matched_fixed_mlp"],
                  "full": ["full_video_saved_mlp"],
                  "no_video": ["text_fixed_mlp"]}
        comparisons = []
        rng = np.random.default_rng(2026)
        for name, controls in groups.items():
            common = set(by_condition["face_fixed_mlp"])
            for condition in controls:
                common &= set(by_condition[condition])
            ids = sorted(common)
            if not ids:
                continue
            differences = {sample_id: by_condition["face_fixed_mlp"][sample_id]
                           - np.mean([by_condition[control][sample_id] for control in controls])
                           for sample_id in ids}
            dialogue_groups = defaultdict(list)
            for sample_id in ids:
                dialogue_groups[dialogues[sample_id]].append(differences[sample_id])
            clusters = list(dialogue_groups.values())
            if len(clusters) > 1:
                boot = [float(np.mean([value for index in rng.integers(0, len(clusters), len(clusters))
                                       for value in clusters[index]])) for _ in range(2000)]
                low, high = np.percentile(boot, [2.5, 97.5])
            else:
                low, high = float("nan"), float("nan")
            comparisons.append({"face_only_minus": name, "samples": len(ids),
                                "dialogues": len(clusters),
                                "mean_delta_p_true": float(np.mean(list(differences.values()))),
                                "bootstrap_ci_low": float(low),
                                "bootstrap_ci_high": float(high)})
        if comparisons:
            with (output / "paired_comparisons.csv").open("w", newline="") as file:
                writer = csv.DictWriter(file, fieldnames=list(comparisons[0]))
                writer.writeheader(); writer.writerows(comparisons)
    print("Matched metrics:", json.dumps(rows, indent=2), flush=True)


def main() -> int:
    parser = args_parser()
    args = parser.parse_args()
    if (args.log_every <= 0 or args.chunk_size <= 0 or args.profile_every < 0
            or (args.epochs is not None and args.epochs <= 0)):
        parser.error("Invalid chunk, log, profile, or epoch interval")
    if args.limit_per_split is not None and args.limit_per_split <= 0:
        parser.error("--limit-per-split must be positive")
    root = args.root.expanduser().resolve()
    meld = (args.meld_root or root / "datasets/meld").expanduser().resolve()
    model_dir = (args.model_dir or root / "models/qwen2_5_vl_3b").expanduser().resolve()
    baseline_dir = (args.baseline_dir or REPO / "experiments/baselines/qwen2_5_vl_3b_shared/outputs").expanduser().resolve()
    output = (args.output_dir or root / "results/qwen2_5_vl_3b_meld_face_only").expanduser().resolve()
    print(f"Verifying environment: {sys.executable}", flush=True)
    environment = verify_qwen_environment()
    face_dependencies = verify_face_dependencies()
    retinaface_weights = verify_retinaface_weights()
    print(f"Verifying MELD at {meld}", flush=True)
    meld_info = verify_meld(meld)
    meld = Path(meld_info["root"])
    print(f"Verifying Qwen2.5-VL-3B at {model_dir}", flush=True)
    model_info = verify_model_3b(model_dir)
    checkpoint_path = baseline_dir / "best_mlp.pt"
    if not checkpoint_path.is_file() or checkpoint_path.stat().st_size < 1_000_000:
        raise FileNotFoundError(f"Saved full-video MLP missing or incomplete: {checkpoint_path}")
    baseline_predictions = load_baseline_predictions(baseline_dir)
    import torch
    baseline_checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if list(baseline_checkpoint["label_names"]) != LABELS:
        raise ValueError(f"Saved full-video MLP label order differs from MELD: {checkpoint_path}")
    if baseline_checkpoint["input_dim"] != 2048:
        raise ValueError(f"Saved full-video MLP has wrong Qwen3B input dimension: {checkpoint_path}")
    if args.verify_only:
        print(json.dumps({"environment": environment, "face_dependencies": face_dependencies,
                          "retinaface_weights": str(retinaface_weights),
                          "meld": meld_info, "model": model_info,
                          "baseline_dir": str(baseline_dir),
                          "baseline_checkpoint_input_dim": baseline_checkpoint["input_dim"],
                          "baseline_prediction_counts": {split: len(rows) for split, rows in baseline_predictions.items()}},
                         indent=2), flush=True)
        return 0

    import cv2
    import numpy as np
    import transformers
    from deepface import DeepFace
    from qwen_vl_utils import process_vision_info
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

    if transformers.__version__ != "4.57.1":
        raise RuntimeError("This pruning boundary is validated with transformers==4.57.1")
    gpu = require_cuda_gpu(torch)
    activate_live_log(output / "run.log")
    print(f"Starting face-token experiment on {gpu}", flush=True)
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["VIDEO_MAX_PIXELS"] = str(args.frame_size**2 * args.max_frames)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    print("Loading RetinaFace detector; its weights must already be available to this node", flush=True)
    retinaface_runtime = build_retinaface_for_allocation(DeepFace)
    config = {"model_id": MODEL_ID, "model_dir": str(model_dir),
              "code_revision": code_revision(),
              "retinaface_weights_sha256": digest(retinaface_weights),
              "retinaface_runtime": retinaface_runtime,
              "model_config_sha256": digest(model_dir / "config.json"),
              "model_weight_bytes": {p.name: p.stat().st_size for p in sorted(model_dir.glob("*.safetensors"))},
              "baseline_checkpoint": str(checkpoint_path),
              "baseline_checkpoint_sha256": digest(checkpoint_path),
              "baseline_dev_predictions_sha256": digest(baseline_dir / "dev_predictions.csv"),
              "baseline_test_predictions_sha256": digest(baseline_dir / "test_predictions.csv"),
              "meld_root": str(meld), "train_csv_sha256": digest(meld / "train/train_sent_emo.csv"),
              "dev_csv_sha256": digest(next(p for p in (meld / "dev_sent_emo.csv", meld / "dev/dev_sent_emo.csv") if p.is_file())),
              "test_csv_sha256": digest(next(p for p in (meld / "test_sent_emo.csv", meld / "test/test_sent_emo.csv") if p.is_file())),
              "fps": args.fps, "min_frames": args.min_frames, "max_frames": args.max_frames,
              "frame_size": args.frame_size, "prompt_style": "emotion_task", "pooling": "last",
              "detector": "retinaface", "overlap_fraction": args.overlap_fraction,
              "control_seeds": CONTROL_SEEDS, "skip_controls": args.skip_controls,
              "limit_per_split": args.limit_per_split, "profile_every": args.profile_every,
              "seed": args.seed, "epochs": args.epochs,
              "versions": {"torch": torch.__version__, "transformers": transformers.__version__,
                           **face_dependencies}}
    output.mkdir(parents=True, exist_ok=True)
    config_path = output / "run_config.json"
    if config_path.is_file():
        previous = json.loads(config_path.read_text())
        if previous != json.loads(json.dumps(config)):
            validate_resume_config(previous, config, output)
    atomic_json(config_path, config)

    index_dir = output / "indexes"
    run_live([sys.executable, "-u", str(REPO / "scripts/data/prepare_meld.py"),
              "--meld-root", str(meld), "--output-dir", str(index_dir)])
    rows_by_split = {}
    for split in SPLITS:
        with (index_dir / f"meld_{split}_index.csv").open(newline="", encoding="utf-8") as file:
            rows = list(csv.DictReader(file))
        if split in baseline_predictions:
            indexed = {row["sample_id"]: row for row in rows}
            unknown = set(baseline_predictions[split]) - set(indexed)
            if unknown:
                raise ValueError(f"Saved {split} baseline has IDs absent from current MELD index: {sorted(unknown)[:5]}")
            rows = [row for row in rows if row["sample_id"] in baseline_predictions[split]]
            for row in rows:
                if row["emotion"] != baseline_predictions[split][row["sample_id"]]["true_label"]:
                    raise ValueError(f"Saved baseline label differs from current MELD: {row['sample_id']}")
        rows_by_split[split] = rows[:args.limit_per_split] if args.limit_per_split else rows
        print(f"{split}: {len(rows_by_split[split])} selected clips", flush=True)

    model = processor = None
    payloads, sample_ids_by_split, usable_by_split, failures, cost_rows = {}, {}, {}, [], []
    for split in SPLITS:
        rows = rows_by_split[split]
        folder = output / "cache" / split
        folder.mkdir(parents=True, exist_ok=True)
        chunks_dir = output / "chunks" / split
        chunks_dir.mkdir(parents=True, exist_ok=True)
        chunk_paths = []
        for start in range(0, len(rows), args.chunk_size):
            end = min(start + args.chunk_size, len(rows))
            chunk_path = chunks_dir / f"{split}_{start:05d}_{end:05d}.pt"
            chunk_paths.append((start, end, chunk_path))
            chunk = None
            if chunk_path.is_file():
                try:
                    chunk = torch.load(chunk_path, map_location="cpu", weights_only=False)
                except Exception as exc:
                    print(f"Could not load {chunk_path}: {exc!r}; rebuilding", flush=True)
            if valid_chunk(chunk, split, start, end, rows) and not chunk["failures"]:
                print(f"Reusing complete chunk {chunk_path} ({end - start} clips)", flush=True)
                continue
            if model is None:
                dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
                print("Loading Qwen for extraction", flush=True)
                model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                    str(model_dir), torch_dtype=dtype, device_map="auto", local_files_only=True,
                    attn_implementation="sdpa",
                ).eval()
                model.requires_grad_(False)
                processor = AutoProcessor.from_pretrained(str(model_dir), local_files_only=True,
                                                          min_pixels=128 * 28 * 28,
                                                          max_pixels=args.frame_size**2)
            print(f"Processing {split} chunk {start}:{end} ({end - start} clips)", flush=True)
            chunk_items, chunk_failures = [], []
            for i in range(start, end):
                row = rows[i]
                cache = folder / f"{row['sample_id']}.pt"
                try:
                    if cache.is_file():
                        item = torch.load(cache, map_location="cpu", weights_only=False)
                        required = {"face"}
                        if split == "dev" and not args.skip_controls and item["face_ordinals"]:
                            required.add("divprune_matched")
                        if item["sample_id"] != row["sample_id"] or item["label"] != row["emotion"]:
                            raise ValueError(f"Cached sample identity mismatch: {cache}")
                        if not required.issubset(item["embeddings"]):
                            raise ValueError(f"Cached conditions missing: {cache}")
                    else:
                        args.current_sample = i + 1
                        print(f"Extracting {split}/{row['sample_id']} ({i + 1}/{len(rows)})", flush=True)
                        item = extract_one(row, split, model, processor, args, torch, np, cv2,
                                           DeepFace, process_vision_info)
                        atomic_torch(cache, item, torch)
                    chunk_items.append(item)
                except Exception as exc:
                    chunk_failures.append({"split": split, "sample_id": row["sample_id"], "error": repr(exc)})
                    print(f"{split}/{row['sample_id']} FAILED: {exc!r}", flush=True)
                if (i + 1) % args.log_every == 0 or i + 1 == len(rows):
                    print(f"{split}: {i + 1}/{len(rows)} visited, "
                          f"{len(chunk_items)} usable in current chunk, "
                          f"{len(chunk_failures)} failures in current chunk", flush=True)
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            chunk = {"split": split, "start": start, "end": end,
                     "expected_sample_ids": [row["sample_id"] for row in rows[start:end]],
                     "items": chunk_items, "failures": chunk_failures}
            if not valid_chunk(chunk, split, start, end, rows):
                raise RuntimeError(f"Chunk coverage mismatch: {chunk_path}")
            atomic_torch(chunk_path, chunk, torch)
            print(f"Saved {chunk_path}: {len(chunk_items)} usable, "
                  f"{len(chunk_failures)} failures", flush=True)
            del chunk_items, chunk
        items = []
        for start, end, chunk_path in chunk_paths:
            chunk = torch.load(chunk_path, map_location="cpu", weights_only=False)
            if not valid_chunk(chunk, split, start, end, rows):
                raise RuntimeError(f"Invalid extraction chunk: {chunk_path}")
            items.extend(chunk["items"])
            failures.extend(chunk["failures"])
            del chunk
        print(f"Merged {len(chunk_paths)} {split} chunks; {len(items)} usable clips", flush=True)
        if not items:
            raise RuntimeError(f"No usable {split} samples; inspect {output / 'run.log'}")
        sample_ids_by_split[split] = [item["sample_id"] for item in items]
        usable_by_split[split] = len(items)
        payload = make_payload(items, "face", torch)
        payloads[(split, "face")] = payload
        atomic_torch(output / f"face_{split}.pt", payload, torch)
        if split == "dev" and not args.skip_controls:
            for name in ["text", "divprune_matched"] + [f"{kind}_{seed}" for kind in ("random", "nonface")
                                                    for seed in CONTROL_SEEDS]:
                if any(name in item["embeddings"] for item in items):
                    payloads[(split, name)] = make_payload(items, name, torch)
        for item in items:
            for condition, cost in item["costs"].items():
                cost_rows.append({"split": split, "sample_id": item["sample_id"],
                                  "dialogue_id": item["dialogue_id"], "condition": condition,
                                  "face_tokens_available": len(item["face_ordinals"]),
                                  "original_visual_tokens": cost["full_visual_tokens"],
                                  "preprocess_seconds": item["preprocess_seconds"],
                                  "face_detection_seconds": item["face_detection_seconds"],
                                  "clip_all_conditions_seconds": item["clip_all_conditions_seconds"],
                                  **cost})
        del items
        with (output / "failures.csv").open("w", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=["split", "sample_id", "error"])
            writer.writeheader(); writer.writerows(failures)

    if model is not None:
        del model
    torch.cuda.empty_cache()
    face_checkpoint = train_face_mlp(output, args, baseline_checkpoint, torch)
    if list(face_checkpoint["label_names"]) != LABELS:
        raise ValueError("Face checkpoint label order differs from MELD")
    for split in SPLITS:
        ids = sample_ids_by_split[split]
        if payloads[(split, "face")]["sample_ids"] != ids:
            raise ValueError(f"{split} face payload is not aligned to extracted clips")
    evaluate(output, payloads, baseline_checkpoint, face_checkpoint,
             baseline_predictions, args, torch, np)
    with (output / "costs.csv").open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(cost_rows[0]))
        writer.writeheader(); writer.writerows(cost_rows)
    summary = {}
    for split in SPLITS:
        for condition in sorted({row["condition"] for row in cost_rows if row["split"] == split}):
            subset = [row for row in cost_rows if row["split"] == split and row["condition"] == condition]
            plain = [row["forward_seconds"] for row in subset if not row["profiler_active"]]
            profiled = [row["counted_operator_tflops"] for row in subset
                        if row["counted_operator_tflops"] is not None]
            summary[f"{split}/{condition}"] = {
                "samples": len(subset),
                "total_original_visual_tokens": sum(row["original_visual_tokens"] for row in subset),
                "total_full_visual_tokens": sum(row["full_visual_tokens"] for row in subset),
                "total_retained_visual_tokens": sum(row["retained_visual_tokens"] for row in subset),
                "mean_original_visual_tokens_per_clip": statistics.mean(
                    row["original_visual_tokens"] for row in subset),
                "mean_retained_visual_tokens_per_clip": statistics.mean(
                    row["retained_visual_tokens"] for row in subset),
                "weighted_token_retention": sum(row["retained_visual_tokens"] for row in subset)
                / sum(row["full_visual_tokens"] for row in subset),
                "mean_forward_seconds_unprofiled": statistics.mean(plain) if plain else None,
                "median_forward_seconds_unprofiled": statistics.median(plain) if plain else None,
                "peak_allocated_gib_max": max(row["peak_allocated_bytes"] for row in subset) / 2**30,
                "profiled_clips": len(profiled),
                "mean_counted_operator_tflops": statistics.mean(profiled) if profiled else None,
            }
    atomic_json(output / "cost_summary.json", summary)
    atomic_json(output / "coverage.json", {
        split: {"selected": len(rows_by_split[split]), "usable": usable_by_split[split],
                "failures": sum(row["split"] == split for row in failures)}
        for split in SPLITS
    })
    print(f"Complete. Metrics: {output / 'matched_metrics.csv'}; cost: {output / 'costs.csv'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
