# Information Horizon

Decoder-depth token information interventions.

| Module | Description |
|---|---|
| [information_horizon.py](information_horizon.py) | Physical decoder-token removal for single-sample, cache-free Qwen2.5-VL. |
| [run_information_horizon.py](run_information_horizon.py) | Information Horizon Eq. 6, adapted to MELD/Qwen2.5-VL. |
| [summarize_information_horizon.py](summarize_information_horizon.py) | Summarize per-sample Information Horizon outputs without inferring a cutoff. |

Direct invocation: `python scripts/information_horizon/information_horizon.py --help`.
Shared algorithm modules expose functions rather than a command-line interface.
