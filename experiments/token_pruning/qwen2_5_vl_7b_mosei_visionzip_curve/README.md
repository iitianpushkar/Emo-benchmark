# VisionZip on Qwen2.5-VL-7B / CMU-MOSEI

This is a video adaptation of the authors' [Qwen2.5-VL VisionZip release](https://github.com/JIA-Lab-research/VisionZip/tree/main/Qwen2_5_VL). That release compresses image tokens; its video branch does not apply VisionZip. Our experiment keeps the existing MOSEI protocol (6 fps, 224 × 224 frames, 4–64 frames, transcript text) and applies the selection to video tokens.

The final full-attention vision block provides attention-received scores and key vectors. Four patch scores/keys are averaged per spatially merged Qwen token, as in the released Qwen implementation. Within each temporal group, the highest-scoring **dominant** tokens are kept. Remaining tokens are assigned by normalized-key similarity to uniformly spaced **contextual** targets, and each target embedding receives the mean embedding of its assigned tokens. The paper's Qwen example retains 65 dominant and 5 contextual tokens per 100 input tokens; here that 13:1 ratio is scaled to every requested final budget. At very low budgets the integer split changes. Temporal groups receive proportional quotas, and merging never crosses groups. Those two decisions are video adaptations, not parts of the released image implementation.

The full vision encoder still runs. The shorter sequence enters the language model. The shared pipeline measures original and retained visual tokens, latency, allocated memory, and partial PyTorch-estimated TFLOPs. It extracts chunked embeddings with resume checkpoints, trains three matched MLP seeds, selects on validation, and leaves test evaluation behind `--test-all`.

From `codes/Emo-benchmark` on the cluster:

```bash
~/hf-env/bin/python cluster/run_mosei_visionzip_curve.py --root ../.. --verify-only
~/hf-env/bin/python cluster/run_mosei_visionzip_curve.py --root ../..
```

Default output: `../../results/qwen2_5_vl_7b_mosei_visionzip_curve/`. Follow `run.log` or the Slurm stdout file for progress. No GPU experiment has been run for this implementation yet.
