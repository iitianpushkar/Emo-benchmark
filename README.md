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


## Environment setup

For Colab, install dependencies at the top of the notebook:

```bash
pip install -r requirements.txt
```

For local development, create a virtual environment first:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

The `.venv/` folder is ignored by Git.

## Prepare MELD indexes

After placing `MELD.Raw` under `dataset/`, build verified split indexes:

```bash
python scripts/prepare_meld.py --meld-root dataset/MELD.Raw --split all
```

For a quick smoke test, limit rows per split:

```bash
python scripts/prepare_meld.py --meld-root dataset/MELD.Raw --split train --limit 100
```


## Extract Qwen video embeddings

Extract frozen Qwen2.5-VL video embeddings before language generation:

```bash
python scripts/extract_qwen_embeddings.py \
  --index-csv outputs/indexes/meld_train_index.csv \
  --output-pt embeddings/qwen2_5_vl_3b/meld_train_100.pt \
  --fps 6 \
  --max-frames 64 \
  --limit 100
```

The extractor uses Qwen as a frozen visual representation model and does not call `model.generate()`.

## Train MLP classifier

After extracting embeddings for train and dev splits, train the lightweight classifier:

```bash
python scripts/train_mlp.py \
  --train-pt embeddings/qwen2_5_vl_3b/meld_train.pt \
  --dev-pt embeddings/qwen2_5_vl_3b/meld_dev.pt \
  --output-dir outputs/qwen2_5_vl_3b \
  --hidden-dim 512 \
  --epochs 50
```

Under the hood, the trainer standardizes embeddings using train-set mean/std, trains a one-hidden-layer MLP, and selects the best checkpoint by dev macro F1.

## Evaluate saved MLP

Evaluate a trained checkpoint on dev or test embeddings without retraining:

```bash
python scripts/evaluate_mlp.py \
  --checkpoint outputs/qwen2_5_vl_3b/best_mlp.pt \
  --embeddings-pt embeddings/qwen2_5_vl_3b/meld_test.pt \
  --output-dir outputs/qwen2_5_vl_3b \
  --split-name test
```

Under the hood, this reloads the MLP, applies the train-set feature mean/std saved in the checkpoint, and writes metrics plus per-sample predictions.
