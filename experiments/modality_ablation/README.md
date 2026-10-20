# Modality Ablation

Video/text contribution controls and evaluation with fixed classifiers.

[All experiments](../README.md)

- [alpha_0_1_frameprune_020_mlp](alpha_0_1_frameprune_020_mlp/README.md): Alpha-zero and alpha-one controls for 20% frame pruning; original inputs retained.
- [alpha_0_both_mlps](alpha_0_both_mlps/README.md): Alpha-zero control evaluated with both the full-baseline and DivPrune classifiers. Original input snapshots are preserved under outputs/inputs.
- [alpha_1_divprune_030_mlp](alpha_1_divprune_030_mlp/README.md): Alpha-one control evaluated with the DivPrune classifier; original inputs retained.
- [video_only_divprune_030](video_only_divprune_030/README.md): Protocol and runnable notebook for video-only DivPrune with two fixed MLPs.
