# Qwen2.5-VL-7B MOSEI seven-class baseline with emotion prompt

This is a new frozen-embedding baseline on the fixed MOSEI subset whose clips
both have an unambiguous seven-class label and succeeded in the earlier
Qwen2.5-VL-7B baseline extraction. Its labels are happy, sad, anger, surprise,
disgust, fear, and neutral. Positive ties were excluded and all-zero scores
became neutral. There are 12,773 train, 1,471 validation, and 3,657 test clips
in the source CSV. New decoder failures, if any, are reported separately.

The prompt asks Qwen to infer exactly one of these seven emotions from the
video and utterance. It lists the choices and cues but never supplies the
true label. The model is frozen; a seven-output MLP is trained on its final
hidden state using train embeddings, selected by validation macro F1, and
evaluated on test. This run extracts **new** embeddings because the prompt
differs from the previous utterance-only baseline.

Copy `mosei_single_label_qwen7b_embedded.csv` from the Mac dataset folder to
`../../datasets/mosei/csv/` on the cluster. From `codes/Emo-benchmark` in a GPU
job, run:

```bash
~/hf-env/bin/python -u cluster/run_mosei_qwen2_5vl7b_baseline.py \
  --root ../.. \
  --single-label-csv ../../datasets/mosei/csv/mosei_single_label_qwen7b_embedded.csv \
  --prompt-style emotion_task \
  --chunk-size 500
```

When the original-MOSEI Decord cache has been built, append
`--decoded-cache-dir ../../datasets/mosei/decoded_qwen_6fps_224_v1/frames`
to reuse its sampled frames. See [the predecode instructions](../../../cluster/PREDECODE_MOSEI.md).

Add `--verify-only` to check the local environment, model, dataset, and subset
counts before using a GPU allocation. The runner uses the existing 224×224,
6-FPS, 4-to-64-frame video protocol, Decord timeout, per-clip progress,
25-clip checkpoints, and resume behavior. It writes to
`../../results/qwen2_5_vl_7b_mosei_single_label_emotion_prompt/` by default:

- `indexes/`: cluster-local clip paths and seven-class targets.
- `chunks/` and `embeddings/`: resumable and merged frozen Qwen embeddings.
- `outputs/`: best MLP checkpoint, metrics, prediction CSVs, and confusion CSVs.
- `cost_summary.json`, `decode_failures.json`, `run.log`, and `summary.json`:
  measured resource use, failures, live log, and combined report.

The existing six-label baseline output directory is separate and is never
reused for this prompted run. For comparison with any other method, intersect
prediction CSVs on `sample_id` because new decoder failures can differ.
