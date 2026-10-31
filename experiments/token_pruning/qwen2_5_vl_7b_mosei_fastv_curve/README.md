# Qwen2.5-VL-7B FastV on CMU-MOSEI

This is a **planned, unrun Qwen adaptation** of [FastV's official ECCV 2024
code](https://github.com/pkunlp-icler/FastV). The official in-place token-drop
path ranks image tokens by the attention they receive from the **last prompt
token** in decoder layer `K-1`, then removes the lowest-ranked image tokens
before layer `K`. Its default `K=2` is used here. The original code targets
LLaVA; this implementation applies the same rule to Qwen2.5-VL video tokens.

Qwen uses multimodal RoPE and grouped-query attention. The score is reconstructed
from the target decoder layer's actual query and key projections, its RoPE
embeddings, and its causal mask while the normal SDPA forward runs. Only video
tokens are ranked. Text and other tokens are kept, and retained video tokens
retain their original order and multimodal RoPE coordinates. The implementation
physically shortens the hidden sequence and later-layer attention masks; it
does not merely mask discarded tokens. Cache-free, batch-one last-token
embedding extraction is supported. At 100% retention, no pruning hook runs.

The shared protocol uses 224 × 224 frames, 6 FPS, 4–64 frames, the six binary
MOSEI emotions, 13 retention ratios from 100% to 5%, and three matched MLP
seeds. Original/retained visual and total sequence-token counts, synchronized
latency, peak PyTorch GPU memory, and partial theoretical forward TFLOPs are
saved through the same measurement path as DivPrune, random, uniform, and
VisPruner. FastV runs the first `K` language-model layers at full sequence
length, so its FLOP savings should differ from pre-LLM pruning at the same
retained-token count. The vision encoder also processes every frame.

From `codes/Emo-benchmark` on the login node:

```bash
~/hf-env/bin/python cluster/run_mosei_fastv_curve.py --root ../.. --verify-only
```

Run a one-clip smoke test in a GPU allocation after an index exists:

```bash
~/hf-env/bin/python cluster/extract_mosei_qwen2_5vl.py \
  --index-csv ../../results/qwen2_5_vl_7b_mosei_divprune_curve/indexes/mosei_valid_index.csv \
  --output-pt ../../results/qwen2_5_vl_7b_mosei_fastv_smoke.pt \
  --model-dir ../../models/qwen --start 0 --limit 1 \
  --retain-ratio 0.3 --pruning fastv --fastv-k 2 --profile-stride 1
```

Then run the resumable validation curve on the GPU node:

```bash
~/hf-env/bin/python cluster/run_mosei_fastv_curve.py --root ../..
```

Watch `../../results/qwen2_5_vl_7b_mosei_fastv_curve/run.log` with `tail -F`.
Chunks checkpoint every 25 new clips by default. Validation scores and costs
are written to `validation_curve.csv` and `validation_curve.png`. Use `--test-all`
only after the validation curve is finished and the test policy is fixed.

Compare with DivPrune at matched measured token budgets and profiled IDs:

```bash
~/hf-env/bin/python cluster/compare_mosei_token_curves.py \
  --divprune-dir ../../results/qwen2_5_vl_7b_mosei_divprune_curve \
  --fastv-dir ../../results/qwen2_5_vl_7b_mosei_fastv_curve \
  --output-csv ../../results/qwen2_5_vl_7b_mosei_fastv_comparison.csv
```
