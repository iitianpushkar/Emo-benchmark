# Qwen2.5-VL-3B Information Horizon on MELD

[Open the Kaggle notebook](notebooks/information_horizon_meld.ipynb).

Attach the updated `benchmark-code` and `meld-dataset` inputs, enable a Kaggle
GPU and Internet, then run every cell in order. The notebook automatically scores
the first 5 available **dev** samples, every projected visual token, and every decoder
depth with the saved full-baseline emotion MLP. No pilot setting or run switch is
needed. It saves one JSON file per completed sample, then exports token scores,
summary tables, and plots. `--score-only` keeps token outputs compact while
retaining seven-class probabilities for the full and no-visual branches.

The exact experiment is very expensive: each sample requires
`1 + depths × (visual tokens + 1)` model forward passes. A single Kaggle session
may not finish. Rerun with the same code, index, checkpoint, and output directory
to resume; across sessions, restore the saved working directory first. Results
are under `/kaggle/working/experiments/information_horizon/`.

The notebook checks for the multi-GPU mask fix reported by the first Kaggle run.
It has been statically validated locally but has not been run to completion on
Kaggle. See the [method guide](../../../docs/information_horizon.md) for the
paper score definition and interpretation limits.
