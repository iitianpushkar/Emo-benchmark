# Qwen2.5-VL-7B-Instruct full baseline on CMU-MOSEI

Status: prepared for cluster execution; no GPU result has been recorded.

Run `python cluster/run_mosei_qwen7b_baseline.py --root ../..` from the
repository root inside an H100 allocation. Use `--model-dir` if the 7B weights
are outside `../../models/qwen`. The launcher refuses the 3B checkpoint.

The target is six independent binary emotions in this order: happy, sad,
anger, surprise, disgust, fear. A score greater than zero is positive. The
released train/valid/test assignment is retained, with 16,326/1,871/4,659
clips respectively. The extractor uses frozen Qwen weights, video plus the
segment transcript, an `utterance_only` prompt, 6 FPS, up to 64 frames resized
to 224 square, and last-layer/last-token embeddings. Chunk files and final
embeddings use separate paths for each split. Extraction must cover every
indexed sample with no recorded errors before classification starts.

The six-output MLP uses train-only standardization and weighted binary cross
entropy. It selects a checkpoint by validation macro F1 at threshold 0.5,
then selects one threshold per emotion on validation. The test split is scored
once with that saved model and those fixed thresholds. Artifacts are written
to `../../results/qwen2_5_vl_7b_mosei/`, including `run_config.json`,
`indexes/`, `chunks/`, `embeddings/`, `outputs/best_mlp.pt`,
`outputs/metrics.json`, and prediction CSVs. A completed run writes
`completed_utc.txt`.
