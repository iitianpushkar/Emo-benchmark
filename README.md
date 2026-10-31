# Emo Benchmark

Experiments on multimodal emotion recognition using frozen vision-language
models and an MLP classifier, with MELD as the current dataset.

## Start here

| Goal | Location |
|---|---|
| Browse experiments and their artifacts | [Experiment catalog](experiments/README.md) |
| Compare completed runs | [Compiled reports](reports/compiled_experiments/README.md) · [Aligned test results](reports/compiled_experiments/ALIGNED_RESULTS.md) |
| Find a command or reusable algorithm | [Script catalog](scripts/README.md) |
| Reproduce extraction, pruning, training, or evaluation | [Workflow guide](docs/workflows.md) |
| Measure token information across decoder layers | [Information Horizon guide](docs/information_horizon.md) |
| Prepare CMU-MOSEI six-emotion runs | [CMU-MOSEI guide](docs/mosei.md) |
| Stage datasets/models and verify a GPU node | [Cluster preflight](cluster/README.md) |
| Find something at its previous location | [Migration guide](docs/repository_layout.md) · [Path map](docs/repository_migration.json) |

## Layout

```text
experiments/                 One directory per experiment/run
  baselines/                 Qwen baseline and encoder comparisons
  token_pruning/             DivPrune and random-token experiments
  frame_pruning/             Diverse/random frame experiments
  modality_ablation/         Video/text contribution and fixed-MLP controls
  conflicts/                 Mismatched video/utterance experiments
  fusion/                    Weighted modality-fusion experiments
  information_horizon/       Per-token probability interventions
scripts/                     Shared tools grouped by purpose
reports/                     Comparisons spanning multiple experiments
data/indexes/                Shared MELD split indexes
dataset/                     Local raw datasets (not redistributed)
archive/kaggle_runs/          Historical downloaded runs and notebooks
docs/                        Methods, workflow recipes, migration, figures
tests/                       Shared algorithm and integration checks
```

Inside an experiment, use `config.yaml`, `notebooks/`, `chunks/`, `embeddings/`,
`outputs/`, `checkpoints/`, and `results/` as needed. `outputs/` holds run logs,
predictions and metrics; `results/` holds curated tables and figures. An existing
run may have only some of these; a directory is not evidence that a run completed.

## Commands

Run commands from the repository root, either by grouped path or by name:

```bash
python -m scripts --list
python -m scripts prepare_meld --meld-root dataset/MELD.Raw --split all
python scripts/classification/train_mlp.py --help
```

Install dependencies in a virtual environment with `pip install -r requirements.txt`.
GPU extraction requires the appropriate model weights and dataset. See the
[workflow guide](docs/workflows.md) for complete experiment commands.

## Tests

```bash
python -m unittest discover -s tests -v
```

The tiny-Qwen integration test requires Transformers. Saved artifacts and
historical metadata are preserved; paths embedded in old checkpoints, downloaded
notebooks, archives, or generated reports may describe their original environment.
Use the migration map to locate the corresponding current local files.
