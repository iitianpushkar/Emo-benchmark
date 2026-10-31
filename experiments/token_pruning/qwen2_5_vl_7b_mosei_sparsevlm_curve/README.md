# Qwen2.5-VL-7B SparseVLM-style curve on CMU-MOSEI

This is a **planned, unrun Qwen2.5-VL adaptation** of the [official SparseVLM
v1.5 implementation](https://github.com/Gumpest/SparseVLMs/tree/v1.5).
The [SparseVLM paper](https://proceedings.mlr.press/v267/zhang25s.html)
**does benchmark Qwen2-VL** (Table 2), as well as LLaVA and video models.
Qwen2-VL in that table is a different backbone from our Qwen2.5-VL-7B.
The publicly released v1.5 branch used here as code reference implements the
LLaVA path; this run does not claim to reproduce its Qwen2-VL benchmark scores.
The reference code selects text raters from embedding similarity, scores video
tokens by rater-to-visual decoder attention, prunes after decoder layers 2, 6,
and 15 (zero-based), and recycles a clustered subset of the discarded tokens.
The adaptation follows these operations on Qwen's video-token positions and
multimodal RoPE, preserving nonvideo tokens and prompt order.

The original v1.5 code hard-codes image-specific retained-token schedules for
LLaVA. This MOSEI curve instead implements the paper's rank rule at each
pruning layer: `floor(lambda * (visual_tokens - rank(P)))` proposed deletions,
where `P` is the selected text-to-visual attention matrix. `lambda` is derived
from the requested final retention as `1 - retention**(1/3)`. The first two
stages cap deletions at the final budget; the last stage is adjusted to make
the **final total visual-token count** exactly match DivPrune. These caps and
the lambda calibration are a benchmark adaptation, so report this as
**rank-adaptive SparseVLM-style pruning at a matched final budget**, not an
exact reproduction of the paper's Qwen2-VL experiment. Per-clip metadata gives
each stage's matrix rank, proposed/applied deletions, direct-kept, recycled,
and output token counts. Recycled
features are placed at their original center positions to preserve Qwen's
causal and multimodal position order.

The shared MOSEI protocol is 224 × 224 frames, 6 FPS, 4–64 frames, six binary
emotions, 100%–5% visual-token retention, and three matched MLP seeds. The
runner reports exact original/final token counts, synchronized latency, GPU
memory, and partial PyTorch counted-operator forward TFLOPs. Early decoder
layers process more than the final token count; compare actual TFLOPs and
latency rather than assuming that final retention equals compute reduction.

From `codes/Emo-benchmark` on the login node:

```bash
~/hf-env/bin/python cluster/run_mosei_sparsevlm_curve.py --root ../.. --verify-only
```

On a GPU compute node, first smoke test one indexed validation clip in a
separate output file:

```bash
~/hf-env/bin/python -u cluster/extract_mosei_qwen2_5vl.py \
  --index-csv ../../results/qwen2_5_vl_7b_mosei_divprune_curve/indexes/mosei_valid_index.csv \
  --output-pt ../../results/qwen2_5_vl_7b_mosei_sparsevlm_smoke.pt \
  --model-dir ../../models/qwen --start 0 --limit 1 \
  --retain-ratio 0.3 --pruning sparsevlm --profile-stride 1
```

Then run the resumable curve:

```bash
~/hf-env/bin/python -u cluster/run_mosei_sparsevlm_curve.py --root ../..
```

Watch `../../results/qwen2_5_vl_7b_mosei_sparsevlm_curve/run.log` with
`tail -F`. Chunks save every 25 new clips by default; restart the same command
to reuse completed work. Compare against the DivPrune curve with:

```bash
~/hf-env/bin/python cluster/compare_mosei_token_curves.py \
  --divprune-dir ../../results/qwen2_5_vl_7b_mosei_divprune_curve \
  --sparsevlm-dir ../../results/qwen2_5_vl_7b_mosei_sparsevlm_curve \
  --output-csv ../../results/qwen2_5_vl_7b_mosei_sparsevlm_comparison.csv
```
