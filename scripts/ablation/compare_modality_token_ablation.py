#!/usr/bin/env python3
"""Compare full embeddings with video-zero or utterance-zero embeddings."""

# Allow both `python scripts/<group>/<file>.py` and `python -m scripts...`.
import sys as _sys
from pathlib import Path as _Path
if __package__ in (None, ""):
    _sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))


from scripts.ablation.compare_visual_token_ablation import main


if __name__ == "__main__":
    main()
