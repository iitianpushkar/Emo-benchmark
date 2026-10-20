# Conflicts

Video/utterance conflict experiments.

| Module | Description |
|---|---|
| [run_meld_conflict_dev.py](run_meld_conflict_dev.py) | Prepare, extract, merge, and evaluate MELD dev conflict pairs. |
| [evaluate_meld_conflicts.py](evaluate_meld_conflicts.py) | Evaluate MLP modality preference on MELD utterance-video conflict pairs. |

Direct invocation: `python scripts/conflicts/run_meld_conflict_dev.py --help`.
Shared algorithm modules expose functions rather than a command-line interface.
