#!/usr/bin/env python3
"""Probe Hub access, stage pinned snapshots, and verify cluster readiness.

This program never starts a model or an experiment. Run ``stage`` on a node
with internet access and ``check`` inside a compute-node allocation.
"""

from __future__ import annotations

import argparse
import csv
import importlib.metadata
import json
import os
import subprocess
import sys
import time
import urllib.request
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path


DATASETS = (
    ("meld", "0xPushkar/MELD", "MELD.Raw"),
    ("mosei", "0xPushkar/CMU-MOSEI", "CMU-MOSEI"),
)
MODELS = (
    ("qwen", "Qwen/Qwen2.5-VL-7B-Instruct"),
    ("internvl3", "OpenGVLab/InternVL3-2B-hf"),
    ("smolvlm2", "HuggingFaceTB/SmolVLM2-2.2B-Instruct"),
    ("dinov2", "facebook/dinov2-base"),
)
DEFAULT_MODEL_NAMES = ("qwen",)
MELD_COUNTS = {"train": 9989, "dev": 1109, "test": 2610}
MOSEI_COUNTS = {"train": 16326, "valid": 1871, "test": 4659}
MELD_VIDEO_DIRS = {
    "train": ("train/train_splits", "train_splits"),
    "dev": ("dev_splits_complete", "dev/dev_splits_complete", "dev_splits", "dev/dev_splits"),
    "test": ("output_repeated_splits_test", "test/output_repeated_splits_test", "test_splits", "test/test_splits"),
}
MELD_CSVS = {
    "train": ("train/train_sent_emo.csv",),
    "dev": ("dev_sent_emo.csv", "dev/dev_sent_emo.csv"),
    "test": ("test_sent_emo.csv", "test/test_sent_emo.csv"),
}


def first_file(root: Path, choices: tuple[str, ...]) -> Path | None:
    return next((root / name for name in choices if (root / name).is_file()), None)


def first_dir(root: Path, choices: tuple[str, ...]) -> Path | None:
    return next((root / name for name in choices if (root / name).is_dir()), None)


def dataset_root(base: Path, name: str) -> Path:
    marker = "csv/mosei_aligned.csv" if name == "mosei" else "train/train_sent_emo.csv"
    label = "CMU-MOSEI" if name == "mosei" else "MELD.Raw"
    candidates = (base, base / label, base / "dataset" / label)
    return next((path for path in candidates if (path / marker).is_file()), base)


def existing_dataset(name: str, stage_root: Path, explicit: Path | None) -> tuple[Path, dict] | None:
    """Find a complete existing dataset before considering any download."""
    base = stage_root / "datasets"
    candidates = [explicit.expanduser().resolve()] if explicit else [base / name]
    if explicit is None and base.is_dir():
        candidates.extend(path for path in sorted(base.iterdir()) if path.is_dir() and path != base / name)
    validator = verify_meld if name == "meld" else verify_mosei
    for candidate in candidates:
        if not candidate.is_dir():
            if explicit is not None:
                raise ValueError(f"Existing {name} directory not found: {candidate}")
            continue
        try:
            validation = validator(candidate)
            return Path(validation["root"]), validation
        except (OSError, ValueError, KeyError) as exc:
            if explicit is not None:
                raise ValueError(f"Existing {name} dataset is incomplete at {candidate}: {exc}") from exc
    return None


def verify_meld(base: Path) -> dict:
    root = dataset_root(base, "meld")
    result = {"root": str(root), "splits": {}, "unexpected_missing": []}
    for split, expected in MELD_COUNTS.items():
        annotation = first_file(root, MELD_CSVS[split])
        video_dir = first_dir(root, MELD_VIDEO_DIRS[split])
        if annotation is None or video_dir is None:
            raise ValueError(f"MELD {split}: missing annotation CSV or video directory under {root}")
        with annotation.open(newline="", encoding="utf-8-sig") as handle:
            rows = list(csv.DictReader(handle))
        if len(rows) != expected:
            raise ValueError(f"MELD {split}: expected {expected} CSV rows, found {len(rows)}")
        missing = []
        for row in rows:
            filename = f"dia{int(row['Dialogue_ID'])}_utt{int(row['Utterance_ID'])}.mp4"
            if not (video_dir / filename).is_file():
                missing.append(filename)
        allowed = {"dia110_utt7.mp4"} if split == "dev" else set()
        unexpected = sorted(set(missing) - allowed)
        result["unexpected_missing"].extend(f"{split}/{name}" for name in unexpected[:20])
        result["splits"][split] = {"rows": len(rows), "missing_clips": len(missing), "missing_examples": missing[:5]}
    if result["unexpected_missing"]:
        raise ValueError(f"MELD has missing annotated clips: {result['unexpected_missing']}")
    return result


def verify_mosei(base: Path, progress_every: int = 0) -> dict:
    root = dataset_root(base, "mosei")
    annotation = root / "csv" / "mosei_aligned.csv"
    if not annotation.is_file():
        raise ValueError(f"MOSEI aligned CSV missing under {root}")
    counts: Counter[str] = Counter()
    missing = []
    started = time.monotonic()
    with annotation.open(newline="", encoding="utf-8-sig") as handle:
        for checked, row in enumerate(csv.DictReader(handle), start=1):
            counts[row["split"]] += 1
            clip = root / "Raw" / row["video_id"] / f"{row['clip_id']}.mp4"
            if not clip.is_file():
                if len(missing) < 20:
                    missing.append(str(clip.relative_to(root)))
            if progress_every and checked % progress_every == 0:
                print(f"MOSEI verification: {checked}/{sum(MOSEI_COUNTS.values())} clip paths checked "
                      f"({time.monotonic() - started:.1f}s elapsed)", flush=True)
    if progress_every:
        print(f"MOSEI verification: {sum(counts.values())} clip paths checked "
              f"({time.monotonic() - started:.1f}s elapsed)", flush=True)
    if dict(counts) != MOSEI_COUNTS:
        raise ValueError(f"MOSEI split counts {dict(counts)} do not match {MOSEI_COUNTS}")
    if missing:
        raise ValueError(f"MOSEI missing clips (first 20): {missing}")
    return {"root": str(root), "splits": dict(counts), "missing_clips": 0}


def verify_model(root: Path, name: str | None = None) -> dict:
    config_path = root / "config.json"
    if not config_path.is_file():
        raise ValueError(f"Model config.json missing: {root}")
    if name == "qwen":
        config = json.loads(config_path.read_text())
        text_config = config.get("text_config", config)
        vision_config = config.get("vision_config", {})
        if (config.get("model_type"), text_config.get("hidden_size"), text_config.get("num_hidden_layers")) != (
            "qwen2_5_vl", 3584, 28
        ) or (vision_config.get("patch_size"), vision_config.get("out_hidden_size")) != (14, 3584):
            raise ValueError(f"Qwen checkpoint at {root} is not Qwen2.5-VL-7B-Instruct")
        required = ("model.safetensors.index.json", "preprocessor_config.json",
                    "tokenizer_config.json")
        absent = [filename for filename in required if not (root / filename).is_file()]
        if absent:
            raise ValueError(f"Qwen model files missing in {root}: {absent}")
        tokenizer_config = json.loads((root / "tokenizer_config.json").read_text())
        if not (root / "chat_template.json").is_file() and not tokenizer_config.get("chat_template"):
            raise ValueError(f"Qwen instruct chat template missing in {root}")
    weights = sorted((*root.glob("*.safetensors"), *root.glob("pytorch_model*.bin")))
    if not weights:
        raise ValueError(f"Model weights missing: {root}")
    too_small = [weight.name for weight in weights if weight.stat().st_size < 1_000_000]
    if too_small:
        raise ValueError(f"Incomplete weights or LFS pointers in {root}: {too_small}")
    for index in (*root.glob("*.safetensors.index.json"), *root.glob("pytorch_model*.bin.index.json")):
        index_data = json.loads(index.read_text())
        shards = set(index_data["weight_map"].values())
        absent = [name for name in shards if not (root / name).is_file()]
        if absent:
            raise ValueError(f"Missing model shards in {root}: {absent[:5]}")
        if name == "qwen":
            expected_bytes = index_data.get("metadata", {}).get("total_size")
            actual_bytes = sum((root / shard).stat().st_size for shard in shards)
            if expected_bytes and actual_bytes < expected_bytes:
                raise ValueError(f"Qwen weight shards are truncated in {root}: {actual_bytes} < {expected_bytes} bytes")
    return {"path": str(root), "weight_files": len(weights)}


def existing_model(name: str, stage_root: Path, explicit: Path | None) -> tuple[Path, dict] | None:
    candidate = explicit.expanduser().resolve() if explicit else stage_root / "models" / name
    if not candidate.is_dir():
        if explicit is not None:
            raise ValueError(f"Existing {name} model directory not found: {candidate}")
        return None
    try:
        return candidate.resolve(), verify_model(candidate, name)
    except (OSError, ValueError, KeyError) as exc:
        if explicit is None and name == "qwen" and (candidate / "config.json").is_file():
            try:
                config = json.loads((candidate / "config.json").read_text())
                text_config = config.get("text_config", config)
                vision_config = config.get("vision_config", {})
                if (config.get("model_type"), text_config.get("hidden_size"), text_config.get("num_hidden_layers")) != (
                    "qwen2_5_vl", 3584, 28
                ) or (vision_config.get("patch_size"), vision_config.get("out_hidden_size")) != (14, 3584):
                    raise ValueError(
                        f"{candidate} contains another Qwen checkpoint; choose a separate Qwen2.5-VL-7B directory"
                    ) from exc
            except json.JSONDecodeError:
                pass
        if explicit is not None:
            raise ValueError(f"Existing {name} model is incomplete at {candidate}: {exc}") from exc
        return None


def parse_existing_models(values: list[str]) -> dict[str, Path]:
    names = {name for name, _ in MODELS}
    models = {}
    for value in values:
        name, separator, raw_path = value.partition("=")
        if not separator or name not in names or not raw_path or name in models:
            raise ValueError(f"Invalid --existing-model {value!r}; use NAME=PATH with NAME in {sorted(names)}")
        models[name] = Path(raw_path)
    return models


def hub_api():
    try:
        from huggingface_hub import HfApi, snapshot_download
    except ImportError as exc:
        raise RuntimeError("Install huggingface_hub in the cluster environment first") from exc
    return HfApi(), snapshot_download


def select_models(names: list[str]) -> tuple[str, ...]:
    available = {name for name, _ in MODELS}
    if names == ["all"]:
        return tuple(name for name, _ in MODELS)
    if not names or len(names) != len(set(names)) or set(names) - available:
        raise ValueError(f"Invalid --models {names!r}; choose from {sorted(available)} or use all")
    return tuple(name for name, _ in MODELS if name in names)


def specs(model_names: tuple[str, ...]):
    yield from ((name, repo, "dataset") for name, repo, _ in DATASETS)
    yield from ((name, repo, "model") for name, repo in MODELS if name in model_names)


def code_revision() -> dict:
    repo_root = Path(__file__).resolve().parents[1]
    try:
        revision = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=repo_root, text=True, stderr=subprocess.DEVNULL
        ).strip()
        dirty = bool(subprocess.check_output(
            ["git", "status", "--porcelain", "--untracked-files=no"],
            cwd=repo_root, text=True, stderr=subprocess.DEVNULL,
        ).strip())
        return {"revision": revision, "dirty_tracked_files": dirty}
    except (OSError, subprocess.CalledProcessError):
        return {"revision": None, "dirty_tracked_files": None}


def hub_probe(model_names: tuple[str, ...]) -> dict:
    api, _ = hub_api()
    result = {}
    for name, repo, repo_type in specs(model_names):
        try:
            info = api.repo_info(repo_id=repo, repo_type=repo_type, files_metadata=False)
            paths = [entry.rfilename for entry in (info.siblings or [])]
            item = {"repo": repo, "revision": info.sha, "files": len(paths), "accessible": True}
            if name == "meld":
                item["video_files"] = sum(path.lower().endswith(".mp4") for path in paths)
                item["layout_ok"] = any(path.endswith("train/train_sent_emo.csv") for path in paths) and item["video_files"] >= 13707
            elif name == "mosei":
                item["video_files"] = sum(path.lower().endswith(".mp4") for path in paths)
                item["layout_ok"] = any(path.endswith("csv/mosei_aligned.csv") for path in paths) and item["video_files"] >= 22856
            else:
                item["layout_ok"] = any(path.endswith("config.json") for path in paths) and any(path.endswith((".safetensors", ".bin")) for path in paths)
            result[name] = item
        except Exception as exc:
            result[name] = {"repo": repo, "accessible": False, "error": f"{type(exc).__name__}: {exc}"}
    try:
        with urllib.request.urlopen("https://github.com/iitianpushkar/Emo-benchmark", timeout=15) as response:
            result["github"] = {"accessible": response.status == 200}
    except Exception as exc:
        result["github"] = {"accessible": False, "error": f"{type(exc).__name__}: {exc}"}
    return result


def environment_probe(require_gpu: bool, gpu_name: str | None = None) -> dict:
    packages = {}
    for name in (
        "torch", "transformers", "huggingface_hub", "accelerate", "qwen-vl-utils",
        "decord", "opencv-python", "pandas", "numpy", "h5py", "scikit-learn",
        "tqdm", "PyYAML", "matplotlib", "Pillow",
    ):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    cuda = {"available": False, "device_count": 0, "names": []}
    try:
        import torch
        cuda["available"] = torch.cuda.is_available()
        cuda["device_count"] = torch.cuda.device_count()
        cuda["names"] = [torch.cuda.get_device_name(i) for i in range(cuda["device_count"])]
    except Exception as exc:
        cuda["error"] = f"{type(exc).__name__}: {exc}"
    name_matches = gpu_name is None or any(gpu_name.lower() in name.lower() for name in cuda["names"])
    return {
        "packages": packages,
        "cuda": cuda,
        "gpu_requirement_met": (not require_gpu or cuda["available"]) and name_matches,
        "requested_gpu_name": gpu_name,
        "node": os.uname().nodename,
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
    }


def check(root: Path, require_gpu: bool, gpu_name: str | None, network: bool,
          model_names: tuple[str, ...]) -> tuple[dict, bool]:
    manifest_path = root / "manifest.json"
    if not manifest_path.is_file():
        raise ValueError(f"Missing {manifest_path}; run stage on a networked node first")
    manifest = json.loads(manifest_path.read_text())
    report = {"stage_root": str(root), "manifest": str(manifest_path), "selected_models": model_names,
              "code": code_revision(), "data": {}, "models": {}, "environment": environment_probe(require_gpu, gpu_name)}
    errors = []
    staged_revision = manifest.get("code", {}).get("revision")
    current_revision = report["code"]["revision"]
    if staged_revision and current_revision and staged_revision != current_revision:
        errors.append(f"Code commit changed since staging: {staged_revision} -> {current_revision}")
    for name, _, _ in DATASETS:
        path = Path(manifest["snapshots"][name]["path"])
        try:
            report["data"][name] = verify_meld(path) if name == "meld" else verify_mosei(path)
        except Exception as exc:
            errors.append(f"{name}: {exc}")
    for name in model_names:
        try:
            path = Path(manifest["snapshots"][name]["path"])
            report["models"][name] = verify_model(path, name)
        except Exception as exc:
            errors.append(f"{name}: {exc}")
    if not report["environment"]["gpu_requirement_met"]:
        errors.append(f"Required CUDA GPU unavailable on this node (name contains {gpu_name or 'any'})")
    missing_packages = [name for name, version in report["environment"]["packages"].items() if version is None]
    if missing_packages:
        errors.append(f"Missing Python packages: {missing_packages}")
    if network:
        report["network"] = hub_probe(model_names)
        if not all(item.get("accessible") and item.get("layout_ok", True) for item in report["network"].values()):
            errors.append("Direct Hub/GitHub access unavailable or a Hub repository lacks required files")
    report["errors"] = errors
    report["ready"] = not errors
    return report, not errors


def stage(root: Path, existing_meld: Path | None, existing_mosei: Path | None,
          existing_models: dict[str, Path], model_names: tuple[str, ...]) -> tuple[dict, bool]:
    api, snapshot_download = hub_api()
    unused = set(existing_models) - set(model_names)
    if unused:
        raise ValueError(f"Existing model paths supplied for unselected models: {sorted(unused)}")
    root.mkdir(parents=True, exist_ok=True)
    (root / "manifest.json").unlink(missing_ok=True)
    manifest = {"created_utc": datetime.now(timezone.utc).isoformat(), "code": code_revision(),
                "selected_models": model_names, "snapshots": {}}
    validation = {}
    for name, repo, repo_type in specs(model_names):
        if repo_type == "dataset":
            explicit = existing_meld if name == "meld" else existing_mosei
            found = existing_dataset(name, root, explicit)
            if found is not None:
                path, validation[name] = found
                print(f"Reusing complete {name} dataset at {path}; no download", flush=True)
                manifest["snapshots"][name] = {
                    "repo": repo, "repo_type": repo_type, "revision": None,
                    "source": "existing_unpinned", "path": str(path),
                }
                continue
        else:
            found = existing_model(name, root, existing_models.get(name))
            if found is not None:
                path, validation[name] = found
                print(f"Reusing complete {name} model at {path}; no download", flush=True)
                manifest["snapshots"][name] = {
                    "repo": repo, "repo_type": repo_type, "revision": None,
                    "source": "existing_unpinned", "path": str(path),
                }
                continue
        info = api.repo_info(repo_id=repo, repo_type=repo_type)
        folder = root / ("datasets" if repo_type == "dataset" else "models") / name
        print(f"Staging {repo_type} {repo}@{info.sha} -> {folder}", flush=True)
        snapshot_download(repo_id=repo, repo_type=repo_type, revision=info.sha, local_dir=folder)
        if repo_type == "dataset":
            validation[name] = verify_meld(folder) if name == "meld" else verify_mosei(folder)
            path = Path(validation[name]["root"])
        else:
            validation[name] = verify_model(folder, name)
            path = folder.resolve()
        manifest["snapshots"][name] = {
            "repo": repo, "repo_type": repo_type, "revision": info.sha,
            "source": "huggingface", "path": str(path),
        }
    # A failed validation leaves the downloaded files for diagnosis, but never a
    # manifest that could make a later compute check appear successful.
    manifest["validation"] = validation
    path = root / "manifest.json"
    temporary = root / "manifest.json.tmp"
    temporary.write_text(json.dumps(manifest, indent=2) + "\n")
    temporary.replace(path)
    return {"manifest": str(path), "snapshots": manifest["snapshots"], "validation": validation, "ready": True}, True


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("probe", "stage", "check"))
    parser.add_argument("--root", type=Path, help="Shared cluster directory for staged datasets, models, and manifest")
    parser.add_argument("--existing-meld", type=Path, help="Use this complete MELD directory instead of downloading it")
    parser.add_argument("--existing-mosei", type=Path, help="Use this complete MOSEI directory instead of downloading it")
    parser.add_argument("--existing-model", action="append", default=[], metavar="NAME=PATH",
                        help="Reuse a complete local model; repeat for qwen, internvl3, smolvlm2, dinov2")
    parser.add_argument("--models", nargs="+", default=list(DEFAULT_MODEL_NAMES),
                        help="Models to probe/stage/check (default: qwen; use all for the full suite)")
    parser.add_argument("--require-gpu", action="store_true", help="Fail check if CUDA is unavailable")
    parser.add_argument("--gpu-name", help="Require a CUDA device name containing this text, e.g. H100")
    parser.add_argument("--network", action="store_true", help="Also test live Hub and GitHub access during check")
    parser.add_argument("--report", type=Path, help="Write the JSON report to this path")
    args = parser.parse_args()
    try:
        model_names = select_models(args.models)
        if args.mode == "probe":
            report = {"network": hub_probe(model_names), "environment": environment_probe(False)}
            report["ready"] = all(item.get("accessible") and item.get("layout_ok", True) for item in report["network"].values())
            success = report["ready"]
        else:
            if args.root is None:
                parser.error("--root is required for stage and check")
            root = args.root.expanduser().resolve()
            report, success = (
                stage(root, args.existing_meld, args.existing_mosei,
                      parse_existing_models(args.existing_model), model_names)
                if args.mode == "stage"
                else check(root, args.require_gpu, args.gpu_name, args.network, model_names)
            )
    except Exception as exc:
        report, success = {"ready": False, "errors": [f"{type(exc).__name__}: {exc}"]}, False
    output = json.dumps(report, indent=2)
    print(output)
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(output + "\n")
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
