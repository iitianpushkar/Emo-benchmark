# Qwen2.5-VL-7B uniform-token control on CMU-MOSEI

This is a **planned, unrun** deterministic coverage control for the
[DivPrune](../qwen2_5_vl_7b_mosei_divprune_curve/README.md) and
[uniform-random](../qwen2_5_vl_7b_mosei_random_curve/README.md) token curves.
It uses the same model, MOSEI splits, transcript, 224 × 224 video frames at
6 FPS, 4–64 frames per clip, 13 retention ratios from 100% to 5%, and three
matched MLP seeds. The six-label validation macro F1 at a fixed threshold
of 0.5 is the main comparison metric.

The selector works on projected video tokens just before language-model
prefill. It groups tokens by Qwen2.5-VL's **temporal mRoPE position ID** and
allocates the exact clip-level token budget across those groups. When the
budget allows, every temporal group retains at least one token. Within each
group, evenly spaced ranks in Morton (Z-order) spatial order provide regular
2D coverage. When the budget is smaller than the number of groups, it keeps
evenly spaced temporal groups. These groups can represent temporal patches,
so they are not necessarily individual decoded frames. Non-video tokens and
the retained tokens' original position IDs remain unchanged. The rule uses no
image features, labels, random seeds, or training.

The vision encoder still processes every frame. Measured per-clip visual and
sequence-token counts, latency, peak PyTorch GPU allocation, and theoretical
forward TFLOPs use the same code path as DivPrune and random pruning. The FLOP
profile runs a separate forward on the same systematically selected clips
per split. PyTorch's counted FLOPs are a partial theoretical estimate, not
measured H100 hardware instructions. The [official Qwen2.5-VL implementation](https://github.com/huggingface/transformers/blob/v4.57.1/src/transformers/models/qwen2_5_vl/modeling_qwen2_5_vl.py)
documents the temporal, height, and width mRoPE coordinates.

## Cluster commands

From `codes/Emo-benchmark`, check staged inputs on the login node:

```bash
python cluster/run_mosei_uniform_curve.py --root ../.. --verify-only
```

Inside an H100 allocation, run the validation curve:

```bash
python cluster/run_mosei_uniform_curve.py --root ../..
```

The resumable output directory is
`../../results/qwen2_5_vl_7b_mosei_uniform_curve/`. Watch progress from the
login node with:

```bash
tail -F ../../results/qwen2_5_vl_7b_mosei_uniform_curve/run.log
```

If the Qwen2.5-VL-7B checkpoint is elsewhere, supply the same `--model-dir`
used for DivPrune and random pruning. Before a full run, check one clip and
the separate FLOP profile on an H100:

```bash
python cluster/extract_mosei_qwen2_5vl.py \
  --index-csv ../../results/qwen2_5_vl_7b_mosei_divprune_curve/indexes/mosei_valid_index.csv \
  --output-pt ../../results/qwen2_5_vl_7b_mosei_uniform_smoke.pt \
  --model-dir ../../models/qwen --start 0 --limit 1 \
  --retain-ratio 0.3 --pruning uniform --profile-stride 1
```

The index must already exist from the baseline or DivPrune preparation.
After the validation comparison and test policy are fixed, add `--test-all`
to score every predeclared retention ratio on held-out test. Do not choose a
ratio using test results.

Compare the three methods only at matched measured token budgets and
profiled sample IDs:

```bash
python cluster/compare_mosei_token_curves.py \
  --divprune-dir ../../results/qwen2_5_vl_7b_mosei_divprune_curve \
  --random-dirs ../../results/qwen2_5_vl_7b_mosei_random_curve_seed42 \
                ../../results/qwen2_5_vl_7b_mosei_random_curve_seed43 \
                ../../results/qwen2_5_vl_7b_mosei_random_curve_seed44 \
  --uniform-dir ../../results/qwen2_5_vl_7b_mosei_uniform_curve \
  --output-csv ../../results/qwen2_5_vl_7b_mosei_token_comparison.csv
```
