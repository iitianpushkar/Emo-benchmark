# CMU-MOSEI transcript baseline

This run uses all 22,856 aligned CMU-MOSEI segments: 16,326 train, 1,871
validation, and 4,659 test. Each of the six CSD emotion scores is positive when
it exceeds zero. The source train/valid/test assignments are preserved, with no
video ID shared across splits. All indexed MP4 clips were present.

`python -m scripts prepare_mosei` creates the indexes, then
`python -m scripts run_mosei_tfidf` trains a TF-IDF word unigram/bigram model and
one class-balanced logistic regression per emotion. Decision thresholds maximize
per-emotion F1 on validation data only. Test predictions are made once with those
thresholds.

| Split | Macro F1 | Micro F1 | Exact subset accuracy |
|---|---:|---:|---:|
| Validation | 0.4159 | 0.4913 | 0.1026 |
| Test | 0.4108 | 0.4867 | 0.1062 |

See [metrics.json](outputs/metrics.json) for each emotion's support, threshold,
precision, recall, F1, and AUROC. [test_predictions.csv](outputs/test_predictions.csv)
contains the test sample IDs, labels, probabilities, and predictions.

This is a transcript-only CPU baseline; it is not a result for any of the frozen
VLM, pruning, ablation, fusion, or information-horizon experiments.
