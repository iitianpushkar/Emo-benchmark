#!/usr/bin/env python3
"""Run the Qwen2.5-VL-7B MELD VisPruner retention curve."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cluster.run_meld_qwen2_5vl7b import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main("vispruner"))
