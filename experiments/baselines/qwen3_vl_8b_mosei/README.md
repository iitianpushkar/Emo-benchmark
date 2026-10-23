# Qwen3-VL-8B-Instruct full baseline on CMU-MOSEI

Status: prepared for cluster execution; no GPU result has been recorded.

Run `python cluster/run_mosei_qwen3vl8b_baseline.py --root ../..` from the
repository root inside an H100 allocation. Use `--model-dir` if the 8B weights
are outside `../../models/qwen`. The launcher refuses older Qwen checkpoints.

The target is six independent binary emotions in this order: happy, sad,
anger, surprise, disgust, fear. A score greater than zero is positive. The
released train/valid/test assignment is retained, with 16,326/1,871/4,659
clips respectively. The extractor uses frozen Qwen weights, video plus the
segment transcript, an `utterance_only` prompt, 6 FPS, 4–64 frames resized
to 448 × 448, and last-layer/last-token embeddings. These shared video settings
are defined in [`cluster/mosei_video_protocol.json`](../../../cluster/mosei_video_protocol.json)
for all roughly 8B VLM comparisons. The saved metadata records each clip's
actual frame count and resized dimensions. Chunk files and final
embeddings use separate paths for each split. Extraction must cover every
indexed sample with no recorded errors before classification starts.

Changing the frame size changes the run configuration; any earlier 224 × 224
artifacts must use a separate output directory. Verify GPU memory on an H100
before submitting the full dataset job, especially for clips near 64 frames.

The six-output MLP uses train-only standardization and weighted binary cross
entropy. It selects a checkpoint by validation macro F1 at threshold 0.5,
then selects one threshold per emotion on validation. The test split is scored
once with that saved model and those fixed thresholds. Artifacts are written
to `../../results/qwen3_vl_8b_mosei/`, including `run_config.json`,
`indexes/`, `chunks/`, `embeddings/`, `outputs/best_mlp.pt`,
`outputs/metrics.json`, and prediction CSVs. A completed run writes
`completed_utc.txt`.
