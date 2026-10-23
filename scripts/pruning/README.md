# Pruning

Reusable token and frame selection algorithms.

| Module | Description |
|---|---|
| [divprune.py](divprune.py) | Paper-faithful max-min diversity selection for visual token pruning. |
| [random_prune.py](random_prune.py) | Deterministic random selection for projected visual-token pruning. |
| [uniform_prune.py](uniform_prune.py) | Deterministic temporal and spatial coverage at an exact token budget. |
| [frame_diversity.py](frame_diversity.py) | Training-free diverse-frame selection with frozen image embeddings. |

Direct invocation: `python scripts/pruning/divprune.py --help`.
Shared algorithm modules expose functions rather than a command-line interface.
