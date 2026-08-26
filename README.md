# Emo Benchmark

Benchmarking vision-language model embeddings for multimodal emotion recognition.

## Dataset layout

Datasets are intentionally not tracked by Git. Place local datasets under `dataset/`:

```text
emo-benchmark/
  dataset/
    MELD.Raw/
      train/
        train_sent_emo.csv
        train_splits/
      dev_sent_emo.csv
      dev_splits_complete/
      test_sent_emo.csv
      output_repeated_splits_test/
```

## Planned pipeline

```text
MELD video clips
  -> sample video frames
  -> extract frozen VLM embeddings before generation/decoder output
  -> pool embeddings per sample
  -> train a lightweight MLP classifier
  -> evaluate accuracy, macro F1, weighted F1, and confusion matrix
```

Generated embeddings and results are also ignored by Git:

```text
embeddings/
outputs/
```
