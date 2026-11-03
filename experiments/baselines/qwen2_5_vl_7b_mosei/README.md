# Qwen2.5-VL-7B-Instruct full baseline on CMU-MOSEI

Status: prepared for cluster execution; no GPU result has been recorded.

Run `python cluster/run_mosei_qwen2_5vl7b_baseline.py --root ../..` from the
repository root inside an H100 allocation. Use `--model-dir` if the 7B weights
are outside `../../models/qwen`. The launcher verifies the model architecture.

The target is six independent binary emotions in this order: happy, sad,
anger, surprise, disgust, fear. A score greater than zero is positive. The
released train/valid/test assignment is retained, with 16,326/1,871/4,659
clips respectively. The extractor uses frozen Qwen weights, video plus the
segment transcript, an `utterance_only` prompt, 6 FPS, 4–64 frames resized
to 224 × 224, and last-layer/last-token embeddings. These shared video settings
are defined in [`cluster/mosei_video_protocol.json`](../../../cluster/mosei_video_protocol.json)
for all roughly 8B VLM comparisons. The saved metadata records each clip's
actual frame count, resized dimensions, token counts, synchronized latency,
and process GPU allocation. About 128 evenly spaced clips per split receive
a separate counted forward for theoretical shape-based TFLOPs. Counting
does not enter the latency measurement. Chunk files and final
embeddings use separate paths for each split. On restart, successful embeddings
and saved decoder failures are reused; only clips without a saved attempt are
processed. The classifier uses available embeddings and reports split coverage.

Changing the frame size changes the run configuration; any earlier 448 × 448
artifacts must use a separate output directory. Verify GPU memory on an H100
before submitting the full dataset job, especially for clips near 64 frames.

The six-output MLP uses train-only standardization and weighted binary cross
entropy. It selects a checkpoint by validation macro F1 at threshold 0.5,
then selects one threshold per emotion on validation. The test split is scored
once with that saved model and those fixed thresholds. Artifacts are written
to `../../results/qwen2_5_vl_7b_mosei/`, including `run_config.json`,
`indexes/`, `chunks/`, `embeddings/`, `outputs/best_mlp.pt`,
`outputs/metrics.json`, `cost_summary.json`, combined `summary.json`, and
prediction CSVs. A run with all clips represented by embeddings writes
`completed_utc.txt`. If decoder failures remain, it writes
`decode_failures.json` and `completed_partial_utc.txt`, and `summary.json`
marks the evaluation scope as successfully decoded clips only. Such metrics
must not be presented as a complete MOSEI test result.

On restart, the launcher requires the original `--chunk-size`. Changes to
the Git revision, GPU allocation, package versions, or resolved paths are
printed but do not stop the run; each changed configuration is recorded in
`resume_configs.json`, while `run_config.json` holds the latest settings.
Individual saved chunks still have their own configuration and sample-ID
checks before reuse.

## Single-label evaluation from saved embeddings

For a **separate seven-class** evaluation, reuse the merged embeddings without
loading Qwen again:

```bash
~/hf-env/bin/python -u cluster/evaluate_mosei_single_label.py \
  --source-dir ../../results/qwen2_5_vl_7b_mosei
```

This trains matched 512-hidden-unit MLPs with seeds 42, 43, and 44 and writes
`single_label/summary.json`, per-seed metrics, prediction CSVs, confusion
matrices, and checkpoints. The target is the largest of the six MOSEI emotion
scores; all-zero clips are neutral. Clips tied at the positive maximum are
excluded from single-label training and evaluation. About 12% of the MOSEI
labels are tied at the maximum. The output records `excluded_ties` separately
from decoder failures. If a fixed-order tie-break sensitivity analysis is
needed, run with `--tie-policy first` in a separate output directory. The
single-label accuracy, macro F1, and weighted F1 are a distinct
task from the six-binary-label metrics above. Decoder failures still reduce
coverage, which is recorded for every split.
