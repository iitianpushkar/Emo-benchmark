# MELD DetectorV2 likely-speaker MLP

This experiment trains the repository's standard Qwen2.5-VL-3B MLP on the
DetectorV2 visual-prompt embeddings. A sample is included only when its
`metadata.speaker_face.status` is `likely_speaker`. The same filter is applied
to train, dev, and test. The checkpoint is selected by dev macro F1.

Run from the repository root with Python, PyTorch, pandas, scikit-learn, and
tqdm installed:

```powershell
python scripts/classification/run_meld_detectorv2_likely_mlp.py `
  --results-dir experiments/detectorv2_likely_speaker/inputs `
  --output-dir experiments/detectorv2_likely_speaker/outputs `
  --epochs 50 --device cpu
```

`inputs` may contain the three `visual_prompt_meld_{train,dev,test}.zip`
archives or the extracted `detectorv2_prompt/meld_{train,dev,test}.pt` files.
The runner saves filtered embeddings, the MLP checkpoint and predictions,
and `aligned_{dev,test}_predictions.csv` plus `aligned_comparison.json`.
The comparison joins predictions by sample ID against the saved full-video
Qwen baseline. Samples without a saved baseline prediction are omitted from
**both** models' paired metrics.

For the Windows run with seed 42, the filtered split sizes were 7,957 train,
897 dev, and 2,096 test. The paired comparison used 893 dev and 2,088 test
samples. Four dev and eight test likely-speaker samples had no saved baseline
prediction.

| Split | Paired samples | DetectorV2 prompt accuracy | Full-video baseline accuracy | DetectorV2 prompt macro F1 | Full-video baseline macro F1 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Dev | 893 | 0.5599 | 0.6226 | 0.4493 | 0.4938 |
| Test | 2,088 | 0.5441 | 0.5934 | 0.3903 | 0.4268 |

The baseline MLP was trained on its original full sample set; only evaluation
is aligned to likely-speaker IDs here. Consequently, the comparison measures
the two existing pipelines on identical evaluation clips, not an isolated
effect of the facial prompt.

## Retraining both MLPs on aligned clips

The stricter paired experiment takes the intersection of DetectorV2
`likely_speaker` IDs and the baseline's clean embedding IDs **in each split**.
It retrains the face-prompt and full-video MLPs with the same architecture,
seed, training IDs, validation IDs, and test IDs:

```powershell
python scripts/classification/run_meld_aligned_likely_mlp.py `
  --face-results-dir experiments/detectorv2_likely_speaker/inputs `
  --output-dir experiments/detectorv2_likely_speaker/aligned_retrain `
  --epochs 50 --device cpu
```

The aligned split sizes are 7,938 train, 893 dev, and 2,088 test. The best
dev macro F1 checkpoints occurred at epoch 34 for the face-prompt MLP and
epoch 27 for the full-video MLP.

| Split | Face-prompt accuracy | Full-video accuracy | Face-prompt macro F1 | Full-video macro F1 |
| --- | ---: | ---: | ---: | ---: |
| Dev | 0.5745 | 0.6181 | 0.4467 | 0.4769 |
| Test | 0.5627 | 0.5967 | 0.3932 | 0.4075 |

Results and paired predictions are in `aligned_retrain/comparison.json` and
`aligned_retrain/aligned_{dev,test}_predictions.csv`. Each model's checkpoint,
metrics, prediction CSVs, and training history are in its `face_prompt` or
`full_video` subdirectory.
