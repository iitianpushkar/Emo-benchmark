#!/usr/bin/env python3
"""Run InternVL3 extraction through the shared resumable chunk runner."""

from __future__ import annotations

import runpy
import sys
from pathlib import Path


def has_argument(name: str) -> bool:
    return name in sys.argv[1:]


script_dir = Path(__file__).resolve().parent
defaults: list[str] = []
if not has_argument("--extractor"):
    defaults.extend(["--extractor", str(script_dir / "extract_internvl3_embeddings.py")])
if not has_argument("--model-id"):
    defaults.extend(["--model-id", "OpenGVLab/InternVL3-2B-hf"])
if not has_argument("--frame-size"):
    defaults.extend(["--frame-size", "384"])

sys.argv[1:1] = defaults
runpy.run_path(str(script_dir / "run_qwen_shared_chunks.py"), run_name="__main__")
