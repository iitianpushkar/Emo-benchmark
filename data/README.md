# Shared prepared data

[indexes/](indexes/) contains the MELD split indexes shared across experiments.
Regenerate them with `python -m scripts prepare_meld --meld-root dataset/MELD.Raw --split all`.
Raw videos remain in `dataset/`; experiment-specific conflict data live under
[experiments/conflicts](../experiments/conflicts/README.md).

For MER2024, first accept the official Hugging Face EULA and authenticate, then run
`python -m scripts download_mer2024` followed by `python -m scripts prepare_mer2024`.
The release contains 5,030 labeled videos. Its default deterministic, stratified
85/15 train/dev indexes contain the 4,813 clips with an English or Chinese
transcript; this is a repository split, not an official challenge split. Pass
`--include-missing-transcripts` to retain all labeled videos for video-only work.

For CMU-MOSEI, run `python -m scripts prepare_mosei --mosei-root dataset/CMU-MOSEI`.
This expects `csv/mosei_aligned.csv` and pre-cut clips under `Raw/<video_id>/<clip_id>.mp4`.
The generated indexes use the source train/valid/test split and six binary targets
defined by each CSD emotion score being greater than zero.
The `emotion_id=0` column exists only for compatibility with the legacy scalar-label
embedding extractors. Use `mosei_mlp.py` to join real targets by `sample_id`; the
extractors' saved scalar `labels` and MELD `label_names` are not MOSEI targets.

To make a MELD-style, seven-class annotation CSV without requiring local video files,
run `python3 scripts/data/create_mosei_single_label_csv.py`. It writes
`dataset/CMU-MOSEI/csv/mosei_single_label.csv`, preserving the original fields
and adding `utterance`, `emotion`, `emotion_id`, `label_status`, and
`max_emotion_score`. The class order is happy, sad, anger, surprise, disgust,
fear, neutral. A unique highest positive score determines the class; positive
ties are omitted; all-zero scores become neutral. The original aligned CSV is
left intact. The output keeps source train/valid/test split IDs, and no video
file check is performed at this annotation-preparation stage.

For a fixed seven-class subset restricted to clips with successfully saved
Qwen2.5-VL-7B baseline embeddings, run:

```bash
python3 scripts/data/filter_mosei_single_label_by_embeddings.py \
  --successful-ids-csv /path/to/successful_extraction_ids_all_splits.csv
```

This writes `dataset/CMU-MOSEI/csv/mosei_single_label_qwen7b_embedded.csv`,
retaining every column of `mosei_single_label.csv` and its train/valid/test
split. It contains 12,773 train, 1,471 validation, and 3,657 test rows for the
current baseline manifest. To create extraction indexes on the cluster, use
`prepare_mosei.py --aligned-csv` pointing to this CSV, `--mosei-root` pointing
to the cluster dataset directory, and a distinct `--output-dir`; the index
builder adds the cluster-local `video_path`. The existing baseline launcher
still prepares its own full six-label indexes, so this subset is for a
separately configured seven-class experiment.
