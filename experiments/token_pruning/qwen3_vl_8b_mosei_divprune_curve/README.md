# Qwen3-VL-8B DivPrune retention curve on CMU-MOSEI

This is a **planned, unrun** experiment. It measures token-level DivPrune at
100, 90, 80, 70, 60, 50, 40, 30, 25, 20, 15, 10, and 5% requested retention.
The 100% condition is extracted by the same pipeline without pruning. The
classifier predicts six independent binary MOSEI emotions; macro F1 across the
six labels is the primary metric. These results must not be compared numerically
with seven-class MELD macro F1.

## Fixed protocol

- Local checkpoint: `Qwen/Qwen3-VL-8B-Instruct`. No model downloads occur in the
  GPU job. The launcher verifies the local checkpoint and complete MOSEI copy.
- Shared [video protocol](../../../cluster/mosei_video_protocol.json): 448 × 448,
  6 FPS, 4–64 frames, segment transcript, and the same prompt and last-token
  embedding as the full baseline.
- DivPrune selects a deterministic max-min cosine-diverse subset of the
  **projected video tokens** at the language-model boundary. It retains their
  original multimodal position IDs and the matching DeepStack visual features.
  Text, timestamps, and vision delimiters remain in the sequence. The vision
  encoder still runs on every input frame, so this experiment measures decoder
  token reduction rather than vision-encoder reduction.
- Train/validation/test split indexes and targets come from `prepare_mosei.py`.
  Every ratio must have complete, aligned sample IDs; extraction failures stop
  the curve. The weighted actual token retention and per-clip extraction time
  are recorded for each split.
- A fresh MLP is trained for each ratio with fixed hyperparameters and seeds
  42, 43, 44. Each run chooses its checkpoint by validation macro F1 at a fixed
  0.5 threshold, then calibrates per-label thresholds on validation. The main
  curve uses the fixed-threshold checkpoint metric. Its error bars show the
  standard deviation across MLP seeds; these seeds do **not** measure variation
  in VLM extraction or DivPrune selection.

## Cluster commands

From `codes/Emo-benchmark` on the login node, validate the already staged
inputs. If the Qwen3 checkpoint is outside `models/qwen`, supply `--model-dir`.

```bash
python cluster/run_mosei_divprune_curve.py --root ../.. --verify-only
```

Then submit this one command inside an allocated H100 job:

```bash
python cluster/run_mosei_divprune_curve.py --root ../..
```

The resumable outputs go to `../../results/qwen3_vl_8b_mosei_divprune_curve/`.
The launcher records model and data hashes, Git revision, requested ratios,
actual token counts, chunk metadata, MLP checkpoints and predictions, and
`validation_curve.csv`, `.json`, and `.png`. It requires the same configuration
when resuming; use a new `--output-dir` after changing code or inputs. A shorter
smoke sweep can be selected explicitly with `--ratios 1 .3 .05`, but it is a
different protocol and must use a different output directory.

Once the full validation curve and selection rule are fixed, evaluate the
predeclared conditions on held-out test with the same command plus `--test-all`.
This reuses the train/validation artifacts, extracts test embeddings, and scores
the saved MLP checkpoints. Report the entire test curve, seed spread, actual
retention, and any failed samples. Do not pick a winning retention after seeing
test scores. Random-token controls and extraction-seed variation are follow-up
experiments, not evidence supplied by this run.

The earlier Kaggle Qwen2.5-VL-3B/MELD results motivated the 30% and 15% points.
They are a different model, dataset, and label task, so they provide no MOSEI
accuracy estimate. In that study the 30% DivPrune test result was promising,
while a combined 15% method decreased development macro F1; this is why this
sweep uses validation-first selection and multiple classifier seeds.
