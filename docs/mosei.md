# CMU-MOSEI six-emotion experiments

The target is six independent binary labels in this order: happy, sad, anger,
surprise, disgust, fear. A CSD score greater than zero is positive. Do not map
these overlapping labels to MELD's single seven-class target. Use the released
train/valid/test assignment, select checkpoints and decision thresholds on valid,
and evaluate test once after selection.

## Prepare data

Run `python -m scripts prepare_mosei --mosei-root dataset/CMU-MOSEI` on a host
holding `csv/mosei_aligned.csv` and the pre-cut `Raw/<video_id>/<clip_id>.mp4`
files. The script verifies unique sample IDs, video-level split isolation, finite
emotion scores, timestamps, and clip existence. It produces
`data/indexes/mosei_{train,valid,test}_index.csv` with six `*_label` columns.
Regenerate these indexes after copying this version of the code to the GPU host;
earlier indexes created on the dataset host lack the compatibility placeholder
columns added for extraction.

The existing Qwen, InternVL3, and SmolVLM2 extractors require scalar
`emotion_id`; the MOSEI index supplies a zero placeholder for extraction only.
Their `.pt` `labels` and `label_names` are therefore placeholders. The MOSEI MLP
loads the six actual targets from the index by `sample_id` and reports missing
embeddings per split.

## Embedding extraction

On a GPU host with model dependencies, start with the full video-and-text Qwen
baseline. Use the same settings for every split and separate output directories
for every condition. The MOSEI-safe `utterance_only` prompt avoids MELD's
seven-class choices:

```bash
python -m scripts run_qwen_shared_chunks \
  --index-csv data/indexes/mosei_train_index.csv \
  --chunks-dir experiments/baselines/qwen2_5_vl_3b_mosei/chunks/train \
  --chunk-prefix train --chunk-size 500 \
  --prompt-style utterance_only --save-dtype float32 \
  --skip-existing-complete
```

Repeat for `valid` and `test`, changing the index, chunk directory, and prefix.
Merge each split with `python -m scripts merge_embedding_chunks` using the
matching `--chunks-dir`, `--output-pt`, and `--pattern 'train_*.pt'` (or valid/test).

The same MOSEI indexes and `--prompt-style utterance_only` can be used with the
InternVL3 and SmolVLM2 chunk runners. Qwen's `--frame-selection`,
`--visual-token-pruning`, and token-ablation options can be used for matched
frame pruning, DivPrune/random controls, combined pruning, and modality
ablation, with distinct directories and identical train/valid/test settings.

## Train and evaluate

```bash
python -m scripts mosei_mlp train \
  --train-pt experiments/baselines/qwen2_5_vl_3b_mosei/embeddings/mosei_train.pt \
  --train-index data/indexes/mosei_train_index.csv \
  --valid-pt experiments/baselines/qwen2_5_vl_3b_mosei/embeddings/mosei_valid.pt \
  --valid-index data/indexes/mosei_valid_index.csv \
  --test-pt experiments/baselines/qwen2_5_vl_3b_mosei/embeddings/mosei_test.pt \
  --test-index data/indexes/mosei_test_index.csv \
  --output-dir experiments/baselines/qwen2_5_vl_3b_mosei/outputs
```

The six-output MLP uses train-only standardization and weighted binary cross
entropy. It selects the checkpoint by validation macro F1 at threshold 0.5, then
selects one threshold per emotion on validation data. The checkpoint saves both
the standardization and thresholds. For a fixed-checkpoint ablation or pruning
comparison, use `python -m scripts mosei_mlp evaluate --checkpoint ...
--embeddings-pt ... --index ... --output-dir ...`; this does not retrain or
retune thresholds.

## Current execution status

The full transcript-only baseline has completed on the dataset host; see
[its result](../experiments/baselines/mosei_tfidf_text/README.md). The VLM
families, including pruning, ablation, fusion, conflicts, and information
horizon, have not been run on MOSEI. The dataset host has CPU-only PyTorch and
an RTX 2050 with 4 GB VRAM, so GPU extraction is deferred to a larger host.
The MELD-specific conflict and information-horizon analyses need MOSEI-specific
target and metric adaptations before they can be claimed as completed.

## Experiment matrix for the GPU host

All runs use the three verified indexes, `utterance_only` prompts, 6 FPS, up to
64 frames, frozen VLM weights, and the six-label MLP above. Keep a separate
directory and extraction config for every row. Reuse the baseline MLP checkpoint
for fixed-checkpoint comparisons; train a matched MLP on each condition where
the MELD experiment trained one.

| Family | Condition | Extractor settings | Status |
|---|---|---|---|
| Baseline | Qwen shared video + text | Qwen chunk runner defaults | Ready to extract |
| Baseline | Qwen video only and text only | `--modality-mode video_only` / `text_only` | Ready to extract |
| Baseline | InternVL3 and SmolVLM2 shared | Corresponding chunk runner | Ready to extract |
| Frame pruning | DINOv2 diversity and seeded random, 20% frames | `--frame-selection diverse` / `random --frame-retain-ratio 0.20` | Ready to extract |
| Token pruning | DivPrune and seeded random, 30% projected visual tokens | `--visual-token-pruning divprune --divprune-retain-ratio 0.30` / random equivalent | Ready to extract |
| Combined pruning | DINOv2 50% frames, DivPrune to 15% of original visual tokens | `--frame-selection diverse --frame-retain-ratio 0.50 --visual-token-pruning divprune --target-combined-visual-retain-ratio 0.15` | Ready to extract |
| Modality ablation | Zero visual tokens, zero utterance tokens, and prompt only | `--visual-token-ablation zero` and/or `--utterance-token-ablation zero` | Ready to extract and evaluate with fixed checkpoint |
| Fusion | Alpha sweep of video-only and text-only embeddings | `fuse_modality_embeddings.py`, then MOSEI MLP evaluation | Ready after both modality embeddings |
| Conflict | Mismatched transcript and video with six-label donor targets | MOSEI-specific pair construction and paired metrics required | Not ported |
| Information horizon | Token interventions with six-label MLP readout | MOSEI-specific multi-label readout required | Not ported |

The existing MELD alpha-sweep wrapper calls a seven-class evaluator; for MOSEI,
fuse one alpha at a time and use `mosei_mlp.py evaluate`. The source MELD
conflict and information-horizon scripts must not be pointed at MOSEI indexes.
