# Qwen2.5-VL-3B Information Horizon on MELD

[Open the Kaggle notebook](notebooks/information_horizon_meld.ipynb).

Attach the updated `benchmark-code` and `meld-dataset` Kaggle inputs, enable GPU
and Internet, and run the cells in order. The code input must include the current
reorganized repository, Information Horizon scripts, and full-baseline checkpoint.

- Default: fixed baseline MLP probabilities, dev split, one-sample pilot.
- Paper readout: set `READOUT = "lm"` for first-answer-token vocabulary probabilities.
- Exhaustive scoring: enable `RUN_SWEEP`, with `SWEEP_DEPTHS = "all"` and
  `SWEEP_TOKENS = 0`. Review the pilot's forward-pass estimate first.
- Outputs: `/kaggle/working/experiments/information_horizon/<RUN_NAME>/`.
- Resume: attach previous run output and set `RESTORE_FROM`; retain identical settings.
- Download: the last cell creates a zip containing results, logs, tables, plots, and provenance.

Transformers is pinned to 4.57.1, matching the decoder-hook integration test.
This notebook has not been GPU-executed locally; there are no precomputed results
in this directory. See the [method guide](../../../docs/information_horizon.md)
for scoring semantics and interpretation limits.
