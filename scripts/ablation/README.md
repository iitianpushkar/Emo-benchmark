# Ablation

Visual/text ablations and fixed-classifier comparisons.

| Module | Description |
|---|---|
| [compare_visual_token_ablation.py](compare_visual_token_ablation.py) | Compare full and zero-visual-token Qwen embeddings with one fixed MLP. |
| [compare_modality_token_ablation.py](compare_modality_token_ablation.py) | Compare full embeddings with video-zero or utterance-zero embeddings. |
| [run_video_only_divprune.py](run_video_only_divprune.py) | Extract the alpha=1 video-only DivPrune condition and evaluate two fixed MLPs. |

Direct invocation: `python scripts/ablation/compare_visual_token_ablation.py --help`.
Shared algorithm modules expose functions rather than a command-line interface.
