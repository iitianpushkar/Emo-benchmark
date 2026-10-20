# Fusion

Modality embedding fusion and alpha sweeps.

| Module | Description |
|---|---|
| [fuse_modality_embeddings.py](fuse_modality_embeddings.py) | Fuse video-only and text-only Qwen LM-space embeddings with alpha weighting. |
| [run_alpha_sweep.py](run_alpha_sweep.py) | Run alpha-weighted modality fusion and evaluate one fixed MLP checkpoint. |
| [summarize_alpha_sweep.py](summarize_alpha_sweep.py) | Summarize alpha sweep MLP metrics. |

Direct invocation: `python scripts/fusion/fuse_modality_embeddings.py --help`.
Shared algorithm modules expose functions rather than a command-line interface.
