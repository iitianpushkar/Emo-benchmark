# Qwen2.5-VL-7B VisPruner on CMU-MOSEI

This is a **planned, unrun adaptation** of the [official ICCV 2025
VisPruner](https://github.com/Theia-4869/VisPruner) selector. The original
implementation uses CLIP CLS-to-patch attention on LLaVA images. Qwen2.5-VL
has no CLS token. Here, each merged video token receives the mean attention
paid to its four constituent patches across queries and heads in Qwen's last
full-attention vision block. Attention is recomputed from that block's actual
Q/K projection and rotary positions while the normal SDPA forward runs. The
selector ranks tokens by this score, keeps 50% of the requested budget as
important tokens, then repeatedly removes the most cosine-similar tokens from
the remaining even-indexed partition, as in the official code. Similarity uses
projected Qwen video features. The important fraction is configurable.

The Qwen-specific attention definition and projected-feature boundary mean
these results should be called **VisPruner adapted to Qwen2.5-VL**, not a
reproduction of the published LLaVA numbers. Vision encoding still processes
all frames. Non-video tokens and original mRoPE positions are preserved.

The shared protocol uses 224 × 224 frames, 6 FPS, 4–64 frames, MOSEI's six
binary emotions, 13 visual-token retention ratios from 100% to 5%, and three
matched MLP seeds. Outputs include original and retained token counts,
validation macro F1, measured latency, PyTorch GPU memory, and partial
theoretical forward TFLOPs. Selection overhead is included in latency and
registered FLOPs. Compare methods only on matched budgets and sample IDs.

From `codes/Emo-benchmark` on the login node:

```bash
python cluster/run_mosei_vispruner_curve.py --root ../.. --verify-only
```

First run a one-clip GPU smoke test after an index has been prepared:

```bash
python cluster/extract_mosei_qwen2_5vl.py \
  --index-csv ../../results/qwen2_5_vl_7b_mosei_divprune_curve/indexes/mosei_valid_index.csv \
  --output-pt ../../results/qwen2_5_vl_7b_mosei_vispruner_smoke.pt \
  --model-dir ../../models/qwen --start 0 --limit 1 \
  --retain-ratio 0.3 --pruning vispruner --important-ratio 0.5 \
  --profile-stride 1
```

Then run the resumable validation curve on an H100 allocation:

```bash
python cluster/run_mosei_vispruner_curve.py --root ../..
```

Watch `../../results/qwen2_5_vl_7b_mosei_vispruner_curve/run.log` with
`tail -F`. Each chunk saves embeddings every 25 new clips by default. The
validation metrics and cost curve appear in `validation_curve.csv` and
`validation_curve.png`. A separate `--test-all` pass evaluates the predefined
ratios on held-out test only after the validation curve is complete.

To compare against an existing DivPrune curve:

```bash
python cluster/compare_mosei_token_curves.py \
  --divprune-dir ../../results/qwen2_5_vl_7b_mosei_divprune_curve \
  --vispruner-dir ../../results/qwen2_5_vl_7b_mosei_vispruner_curve \
  --output-csv ../../results/qwen2_5_vl_7b_mosei_vispruner_comparison.csv
```
