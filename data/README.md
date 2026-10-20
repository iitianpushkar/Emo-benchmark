# Shared prepared data

[indexes/](indexes/) contains the MELD split indexes shared across experiments.
Regenerate them with `python -m scripts prepare_meld --meld-root dataset/MELD.Raw --split all`.
Raw videos remain in `dataset/`; experiment-specific conflict data live under
[experiments/conflicts](../experiments/conflicts/README.md).
