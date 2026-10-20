# Repository organization and migration

The repository is organized along two axes:

- **Experiment directories** collect the notebooks, configurations, embeddings,
  predictions, checkpoints, and figures needed to understand one run.
- **Script groups** hold shared implementations used by multiple experiments.

The [experiment catalog](../experiments/README.md) and
[script catalog](../scripts/README.md) are the main entry points.

## What moved

| Previous location | Current location |
|---|---|
| Flat `scripts/*.py` | `scripts/<purpose>/*.py` |
| `outputs/indexes/` | `data/indexes/` |
| `embeddings/<run>/`, `outputs/<run>/`, `results/<run>/` | `experiments/<family>/<run>/<artifact_type>/` |
| Root notebooks and configuration | Their experiment directory |
| `results/compiled_experiments/` | `reports/compiled_experiments/` |
| `outputs/kaggle_runs/` | `archive/kaggle_runs/` |
| Checkpoint under `assets/` | Token-pruning experiment's `checkpoints/` |
| Long root README recipes | [Workflow guide](workflows.md) |

[repository_migration.json](repository_migration.json) maps every moved source
path to its destination. Directory mappings apply to their descendants. Resolve
an old path by applying the longest matching old prefix once.

Scripts support direct execution and package imports. The shared script registry
resolves subprocess targets across groups, independent of working directory.
CLI data paths remain relative to the caller, so use the repository root for
commands in documentation. Alternatively use absolute input/output paths.

```bash
python -m scripts --list
python -m scripts train_mlp --help
python scripts/classification/train_mlp.py --help
```

Old flat script filenames are no longer entry points; update external jobs using
the path map or the `python -m scripts <name>` dispatcher.

## Data and provenance

Raw MELD videos remain under `dataset/` so absolute video paths in split indexes
continue to work. Binary tensors, checkpoints, generated CSV/JSON results, and
archived downloads were moved without modifying their contents. Embedded paths
in these files describe the original run and are intentionally retained.

Downloaded Kaggle notebooks/logs are historical snapshots. Active notebooks have
updated repository references; their external `/kaggle/input` locations and scratch
output conventions remain environment-specific. Downloaded zip member paths in
`compile_aligned_results.py` remain unchanged because they refer to external
archives, not this repository's layout.

The reorganization does not rerun experiments or change reported metrics.
Use separate run directories for future experiments, and avoid mixing artifacts
from different sampling, pruning, model, or classifier configurations.
