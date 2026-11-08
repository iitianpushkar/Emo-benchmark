#!/usr/bin/env python3
"""Predecode original MELD clips with parallel, resumable Decord workers."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cluster.predecode_mosei import main  # noqa: E402


if __name__ == "__main__":
    raise SystemExit(main("meld"))
