# Data

Prepare MELD indexes and conflict datasets.

| Module | Description |
|---|---|
| [prepare_meld.py](prepare_meld.py) | Build clean MELD split indexes with verified video paths. |
| [create_meld_conflict_dataset.py](create_meld_conflict_dataset.py) | Create deterministic MELD utterance-video conflict pair datasets. |

Direct invocation: `python scripts/data/prepare_meld.py --help`.
Shared algorithm modules expose functions rather than a command-line interface.
