# Classification

Train and evaluate the shared emotion MLP.

| Module | Description |
|---|---|
| [train_mlp.py](train_mlp.py) | Train an MLP emotion classifier on frozen VLM embeddings. |
| [evaluate_mlp.py](evaluate_mlp.py) | Evaluate a saved MLP checkpoint on extracted VLM embeddings. |

Direct invocation: `python scripts/classification/train_mlp.py --help`.
Shared algorithm modules expose functions rather than a command-line interface.
