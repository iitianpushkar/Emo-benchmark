# Data

Prepare MELD indexes and conflict datasets.

| Module | Description |
|---|---|
| [prepare_meld.py](prepare_meld.py) | Build clean MELD split indexes with verified video paths. |
| [download_mer2024.py](download_mer2024.py) | Download the gated MER2024 labeled videos and metadata. |
| [prepare_mer2024.py](prepare_mer2024.py) | Build deterministic, transcript-complete MER2024 train/dev indexes with verified video paths. |
| [export_mosei_csd.py](export_mosei_csd.py) | Export CMU-MOSEI labels, timestamped words, and reconstructed segment transcripts from CSD to CSV. |
| [merge_mosei_labels.py](merge_mosei_labels.py) | Merge processed CMU-MOSEI splits and transcripts with CSD timestamps and emotion labels. |
| [create_meld_conflict_dataset.py](create_meld_conflict_dataset.py) | Create deterministic MELD utterance-video conflict pair datasets. |

Direct invocation: `python scripts/data/prepare_meld.py --help`.
Shared algorithm modules expose functions rather than a command-line interface.
