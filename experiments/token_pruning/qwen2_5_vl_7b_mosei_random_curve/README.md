# Qwen2.5-VL-7B random-token control on CMU-MOSEI

This is a **planned, unrun** control for the [DivPrune retention curve](../qwen2_5_vl_7b_mosei_divprune_curve/README.md).
It uses the same Qwen2.5-VL-7B checkpoint, MOSEI train/validation/test
indexes, 224 × 224 video frames at 6 FPS, 4–64 frames per clip, transcript,
last-token embedding, and matched six-label MLP training. It tests the same
13 requested projected video-token retention ratios from 100% to 5%.

At the language-model boundary, the control samples exactly the same number
of video tokens as DivPrune, uniformly without replacement. Non-video tokens
and retained tokens' original multimodal position IDs remain unchanged. The
selection is deterministic for each pair of base random seed and sample ID;
it does not depend on chunk boundaries, process order, or restarts. The 100%
condition has no pruning. As with DivPrune, the vision encoder processes all
frames. This is a decoder-token control, not a frame-pruning control.

Every ratio records actual per-clip visual and total token counts, latency,
GPU allocation, and theoretical forward TFLOPs on the same systematically
sampled clips as DivPrune. FLOP counting uses the same PyTorch operator
formulas and excludes unsupported operations; it is not a hardware counter.
The profile forward uses the **same per-sample random seed** as the timed
forward. A separate matched MLP is trained at seeds 42, 43, and 44 for each
ratio. The primary validation metric is six-label macro F1 at a fixed 0.5
threshold; test remains held out until the validation comparison is fixed.

## Cluster commands

From `codes/Emo-benchmark`, validate staged inputs on the login node:

```bash
python cluster/run_mosei_random_curve.py --root ../.. --verify-only
```

Before the full curve, a one-clip H100 smoke run can check the random hook
and its separate FLOP-profile forward. Use the already prepared validation
index from the DivPrune or baseline output directory:

```bash
python cluster/extract_mosei_qwen2_5vl.py \
  --index-csv ../../results/qwen2_5_vl_7b_mosei_divprune_curve/indexes/mosei_valid_index.csv \
  --output-pt ../../results/qwen2_5_vl_7b_mosei_random_smoke.pt \
  --model-dir ../../models/qwen --start 0 --limit 1 \
  --retain-ratio 0.3 --pruning random --random-seed 42 --profile-stride 1
```

In an H100 allocation, run independent extraction seeds. Each has its own
resumable directory and live `run.log`:

```bash
python cluster/run_mosei_random_curve.py --root ../.. --random-seed 42
python cluster/run_mosei_random_curve.py --root ../.. --random-seed 43
python cluster/run_mosei_random_curve.py --root ../.. --random-seed 44
```

If the Qwen2.5-VL-7B checkpoint is outside `../../models/qwen`, pass its
`--model-dir` to every command. A single seed is suitable for a smoke run;
reporting the random-control variation requires the three independent
extraction seeds. The MLP seeds quantify classifier variation separately.
Checkpointed embeddings survive restarts as described in the
[cluster run guide](../../../cluster/README.md).

After all validation curves are complete, compare them at exactly matched
visual-token counts and FLOP-profile sample IDs:

```bash
python cluster/compare_mosei_token_curves.py \
  --divprune-dir ../../results/qwen2_5_vl_7b_mosei_divprune_curve \
  --random-dirs ../../results/qwen2_5_vl_7b_mosei_random_curve_seed42 \
                ../../results/qwen2_5_vl_7b_mosei_random_curve_seed43 \
                ../../results/qwen2_5_vl_7b_mosei_random_curve_seed44 \
  --output-csv ../../results/qwen2_5_vl_7b_mosei_random_comparison.csv
```

The comparison reports the difference of validation means and the random
variation across extraction seeds. The MLP spread is reported separately;
MLP runs sharing one extraction are not independent extraction replicates.
Add `--uniform-dir ../../results/qwen2_5_vl_7b_mosei_uniform_curve` to include
the [deterministic uniform control](../qwen2_5_vl_7b_mosei_uniform_curve/README.md)
in the same matched-budget comparison.
After fixing the test policy, add `--test-all` to each random run to evaluate
the held-out test set. Do not select a retention ratio using test metrics.
