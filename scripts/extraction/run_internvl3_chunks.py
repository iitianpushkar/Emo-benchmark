#!/usr/bin/env python3
"""Run InternVL3 extraction through the shared resumable chunk runner."""

from __future__ import annotations

# Allow both `python scripts/<group>/<file>.py` and `python -m scripts...`.
import sys as _sys
from pathlib import Path as _Path
if __package__ in (None, ""):
    _sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))

from scripts._paths import resolve_script

import runpy
import sys
from pathlib import Path


def has_argument(name: str) -> bool:
    return name in sys.argv[1:]


defaults: list[str] = []
if not has_argument("--extractor"):
    defaults.extend(["--extractor", str(resolve_script("extract_internvl3_embeddings.py"))])
if not has_argument("--model-id"):
    defaults.extend(["--model-id", "OpenGVLab/InternVL3-2B-hf"])
if not has_argument("--frame-size"):
    defaults.extend(["--frame-size", "384"])

sys.argv[1:1] = defaults
runpy.run_path(str(resolve_script("run_qwen_shared_chunks.py")), run_name="__main__")
