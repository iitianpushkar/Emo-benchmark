# Shared prepared data

[indexes/](indexes/) contains the MELD split indexes shared across experiments.
Regenerate them with `python -m scripts prepare_meld --meld-root dataset/MELD.Raw --split all`.
Raw videos remain in `dataset/`; experiment-specific conflict data live under
[experiments/conflicts](../experiments/conflicts/README.md).

For MER2024, first accept the official Hugging Face EULA and authenticate, then run
`python -m scripts download_mer2024` followed by `python -m scripts prepare_mer2024`.
The release contains 5,030 labeled videos. Its default deterministic, stratified
85/15 train/dev indexes contain the 4,813 clips with an English or Chinese
transcript; this is a repository split, not an official challenge split. Pass
`--include-missing-transcripts` to retain all labeled videos for video-only work.
