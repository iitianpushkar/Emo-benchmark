# Qwen2.5-VL-7B MOSEI seven-class baseline with emotion prompt

This is a frozen-embedding baseline on MOSEI clips with an unambiguous
seven-class label. Its labels are happy, sad, anger, surprise, disgust, fear,
and neutral. Positive ties are excluded and all-zero scores become neutral.
With the predecoded cache, the runner intersects the full single-label CSV
with `successful_decodes.csv`, verifies each cached clip, and extracts only
that intersection. The resulting counts are printed before GPU work.

The prompt asks Qwen to infer exactly one of these seven emotions from the
video and utterance. It lists the choices and cues but never supplies the
true label. The model is frozen; a seven-output MLP is trained on its final
hidden state using train embeddings, selected by validation macro F1, and
evaluated on test. This run extracts **new** embeddings because the prompt
differs from the previous utterance-only baseline.

Create `../../datasets/mosei/csv/mosei_single_label.csv` from the original
annotations if it is not already on the cluster:

```bash
~/hf-env/bin/python scripts/data/create_mosei_single_label_csv.py \
  --source ../../datasets/mosei/csv/mosei_aligned.csv \
  --output ../../datasets/mosei/csv/mosei_single_label.csv
```

From `codes/Emo-benchmark` in a GPU job, run:

```bash
~/hf-env/bin/python -u cluster/run_mosei_qwen2_5vl7b_baseline.py \
  --root ../.. \
  --single-label-csv ../../datasets/mosei/csv/mosei_single_label.csv \
  --prompt-style emotion_task \
  --decoded-cache-dir ../../datasets/mosei/decoded_qwen_6fps_224_v1 \
  --chunk-size 500
```

The runner writes the chosen input rows to
`../../results/qwen2_5_vl_7b_mosei_single_label_emotion_prompt_cache_success/cached_single_label_subset.csv`.
The full single-label CSV supplies the ground truth; the cache success manifest
supplies eligibility. The older `mosei_single_label_qwen7b_embedded.csv` is not
used in this run. See [the predecode instructions](../../../cluster/PREDECODE_MOSEI.md).

Add `--verify-only` to check the local environment, model, dataset, and subset
counts before using a GPU allocation. The runner uses the existing 224×224,
6-FPS, 4-to-64-frame video protocol, Decord timeout, per-clip progress,
25-clip checkpoints, and resume behavior. It writes to
`../../results/qwen2_5_vl_7b_mosei_single_label_emotion_prompt_cache_success/` by default when the cache is supplied:

- `indexes/`: cluster-local clip paths and seven-class targets.
- `chunks/` and `embeddings/`: resumable and merged frozen Qwen embeddings.
- `outputs/`: best MLP checkpoint, metrics, prediction CSVs, and confusion CSVs.
- `cost_summary.json`, `decode_failures.json`, `run.log`, and `summary.json`:
  measured resource use, failures, live log, and combined report.

The existing six-label baseline output directory is separate and is never
reused for this prompted run. For comparison with any other method, intersect
prediction CSVs on `sample_id` because new decoder failures can differ.
