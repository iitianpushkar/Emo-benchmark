# Token Pruning

Projected visual-token pruning with DivPrune and random controls.

[All experiments](../README.md)

- [qwen2_5_vl_3b_divprune_030](qwen2_5_vl_3b_divprune_030/README.md): Reusable 30% DivPrune classifier checkpoint. Other pruning runs remain in external result archives referenced by compiled reports.
- [Qwen2.5-VL-7B DivPrune MOSEI curve](qwen2_5_vl_7b_mosei_divprune_curve/README.md): Planned 100%–5% token-retention sweep.
- [Qwen2.5-VL-7B random-token MOSEI curve](qwen2_5_vl_7b_mosei_random_curve/README.md): Matched random control with independent extraction seeds.
- [Qwen2.5-VL-7B uniform-token MOSEI curve](qwen2_5_vl_7b_mosei_uniform_curve/README.md): Deterministic temporal and spatial coverage control.
- [Qwen2.5-VL-7B VisPruner MOSEI curve](qwen2_5_vl_7b_mosei_vispruner_curve/README.md): Qwen-adapted vision-attention plus diversity selector with matched retention and cost measurements.
- [Qwen2.5-VL-7B VisionZip MOSEI curve](qwen2_5_vl_7b_mosei_visionzip_curve/README.md): Official Qwen2.5 image selection adapted to video with dominant tokens, contextual merging, and matched budgets.
- [Qwen2.5-VL-7B FastV MOSEI curve](qwen2_5_vl_7b_mosei_fastv_curve/README.md): Last-query attention pruning after two Qwen decoder layers, with real token dropping and matched cost measurements.
- [Qwen2.5-VL-7B SparseVLM-style MOSEI curve](qwen2_5_vl_7b_mosei_sparsevlm_curve/README.md): Rank-adaptive text-guided decoder pruning with clustered token recycling and matched final budgets.
