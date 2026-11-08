#!/usr/bin/env python3
"""Make per-layer, every-frame attention overlays for one cached MOSEI clip."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cluster.visualize_meld_full_attention import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main("mosei"))
