#!/usr/bin/env python3
"""Run Qwen2.5-VL-7B VisionZip video adaptation on CMU-MOSEI."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cluster.run_mosei_divprune_curve import main  # noqa: E402


if __name__ == "__main__":
    sys.exit(main(method="visionzip"))
