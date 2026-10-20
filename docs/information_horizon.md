# Information Horizon on MELD

Reference: [Wang et al., CVPR 2026, Section 3.2 / Eq. 6](https://arxiv.org/html/2512.07580v2).
Official implementation: https://github.com/YahongWang1/Information-Horizon

For each sample, decoder depth d, and visual token k, compute

    I_d(k) = P(y | only visual token k retained at d)
             - P(y | all visual tokens removed at d)

Both branches run the unmodified multimodal prefix through d completed decoder
blocks. They retain every text/special token and its existing hidden state.
Depth 0 means before the first decoder block. Removal is physical sequence
compaction, with causal masks and original rotary positions sliced consistently
in all remaining blocks. Negative scores are retained. This is not leave-one-out
from the full sequence, not zeroing embeddings, and not a text-only input run.
Text states at later depths already contain visual information.

Two readouts are supported:

* `--readout lm`: the paper's first-answer-token probability from the full
  vocabulary softmax. Each emotion name is tokenized and its first token used;
  these seven values need not sum to one and are not full-label likelihoods.
* `--readout mlp`: an explicit adaptation using the frozen emotion classifier
  and its saved standardization. Seven probabilities sum to one. Match the
  original model, prompt, sampling and pooling configuration. No retraining.

The default is MLP for this benchmark. The intervention/score follows the paper;
MELD video and the MLP readout are not a reproduction of its image-VQA results.
The runner recomputes the full baseline with identical inputs; existing prediction
CSVs are reference checks, not substitutes for the depth-specific absent branch.

## Kaggle notebook

Use the [MELD Information Horizon notebook](../experiments/information_horizon/qwen2_5_vl_3b_meld/notebooks/information_horizon_meld.ipynb) for Kaggle setup, automatic scoring of the first 2 dev samples at every visual token and decoder depth, per-sample resume, plots, and downloads.

## Run a pilot

```bash
python scripts/information_horizon/run_information_horizon.py \
  --index-csv data/indexes/meld_dev_index.csv \
  --checkpoint experiments/baselines/qwen2_5_vl_3b_shared/outputs/best_mlp.pt \
  --output-dir experiments/information_horizon/information_horizon_pilot/outputs \
  --depths 0,8,16,24 --max-tokens 8 --limit 1

python scripts/information_horizon/summarize_information_horizon.py \
  --input-dir experiments/information_horizon/information_horizon_pilot/outputs
```

For exhaustive scoring use `--depths all --max-tokens 0`. For the paper's LM
readout replace `--checkpoint ...` with `--readout lm` and use a fresh directory.
Requires the project's extraction dependencies and Qwen weights. Eager attention
is intentional; FlashAttention and KV caching are unsupported. Start with one
sample: N visual tokens and D depths require 1 + D*(N+1) forward passes. The
current implementation recomputes vision/prefix features for each intervention;
it favors a simple reference implementation over speed. A positive max-tokens
selects evenly spaced ordinals and is a pilot, not exhaustive scoring.

`--score-only` saves only each token's ground-truth score and present probability, reducing storage for exhaustive analysis. The full seven-class per-token arrays remain the default without that flag.

Each completed sample is saved atomically to a JSON file named by a sample-ID
hash. Files include class order, full probabilities, no-visual probabilities per
depth, original sequence positions, and signed/absolute mean and variance of
ground-truth scores. By default each token also has seven-class probabilities and
signed differences; with `--score-only`, each token has its ground-truth score and
present probability instead.
`config.json` records settings, software versions, and checkpoint/index hashes.
`--resume` requires identical settings and skips completed samples; an interrupted
sample is recomputed. Invalid/failed samples stop the run rather than disappearing
silently. The summary uses equal sample weighting; LM correctness is only an
argmax over the seven first-token scores, not generated-answer accuracy.

A near-zero signed mean alone does not establish a horizon: positive and negative
scores can cancel. Inspect absolute scores, variance, and the full-versus-absent
prediction differences together. Ground-truth-driven ranking is an oracle analysis;
using held-out labels to select deployment tokens would leak labels.
