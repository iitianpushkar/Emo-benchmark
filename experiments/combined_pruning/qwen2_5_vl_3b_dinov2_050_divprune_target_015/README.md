# DINOv2 50% Frame Selection + Adaptive DivPrune Token Selection

[Open the Kaggle notebook](notebooks/combined_frame_token_pruning_015.ipynb).

The notebook uses the existing DINOv2-base centroid-seeded max-min cosine
selector to keep about 50% of candidate frames, then applies DivPrune to
projected Qwen visual tokens before decoder block 0. Its adaptive target keeps
about 15% of the **unpruned Qwen visual-token count** per sample. This needs a
second Qwen preprocessing pass because selecting fewer frames can change the
number of projected tokens per frame. The notebook records the measured
retention, extraction errors, sample coverage, and aligned dev/test metrics.

It trains a matched MLP on the combined-pruning embeddings and also evaluates
the unchanged full-baseline MLP. Attach the current `benchmark-code` and
`meld-dataset` Kaggle inputs, enable GPU and Internet, and run all cells. The
extraction runs in separate 500-sample processes, saves progress every 25
samples within a chunk, skips completed chunks on rerun, and merges the chunk
files before classification. No results have been run locally for this combined
condition.
