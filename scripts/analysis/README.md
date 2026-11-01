# Analysis

Cross-experiment aggregation and sample alignment.

| Module | Description |
|---|---|
| [compile_aligned_results.py](compile_aligned_results.py) | Recompute MELD test metrics on one exact sample-ID intersection. |
| [qwen_decoder_attention.py](qwen_decoder_attention.py) | Capture head-averaged attention from Qwen's final prompt-token query without storing full attention matrices. |
| [visualize_mosei_decoder_attention.py](visualize_mosei_decoder_attention.py) | Produce per-layer CMU-MOSEI temporal, spatial, and frame-overlay attention heatmaps. |

Direct invocation: `python scripts/analysis/compile_aligned_results.py --help`.
Shared algorithm modules expose functions rather than a command-line interface.
