# Video-only (alpha=1) with 30% DivPrune

This experiment runs Qwen2.5-VL-3B with video and the emotion-task prompt,
without utterance text. DivPrune retains 30% of projected visual tokens.
Both existing classifiers evaluate the same fresh dev/test embeddings:
the full video+text baseline MLP and the video+text 30%-DivPrune MLP.
There is no training split extraction or classifier retraining.

Run on the GPU environment containing MELD videos, the repository dependencies,
and both checkpoints. Index CSV `video_path` values must resolve there.
From the repository directory:

```bash
python scripts/ablation/run_video_only_divprune.py \
  --index-dir /path/to/data/indexes \
  --baseline-checkpoint /path/to/qwen2_5_vl_3b_shared/best_mlp.pt \
  --divprune-checkpoint /path/to/qwen2_5_vl_3b_divprune_070/new_mlp/best_mlp.pt \
  --output-root /kaggle/working/video_only_divprune_030
```

Use the checkpoint from `Downloads/results/divprune_exp_0.3`: its internal
`070` directory name is misleading; the archived extraction metadata confirms
`divprune_retain_ratio=0.3`. Do not use the checkpoint from `divprune_exp_0.7`.
Supply actual environment paths in place of `/path/to/...`.

Add `--dry-run` to inspect commands. Re-running resumes completed chunks.
The output directory contains merged embeddings, per-checkpoint dev/test
metrics and predictions, `summary.csv`, and `run_manifest.json`.
Extraction errors and successful sample counts are reported; comparisons with
older experiments should align sample IDs if their extraction failures differ.

The previous archived alpha=1 run used `normalized_before_fusion=False`.
Accordingly, this experiment evaluates the direct video-only embeddings;
each MLP applies its own saved training mean and standard deviation.
Sampling matches the previous setup: 6 FPS, up to 64 frames, minimum 4 frames,
224 frame size, last-token pooling at layer -1, and the assistant generation marker.
