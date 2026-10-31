# Classification

Train and evaluate the shared emotion MLP.

| Module | Description |
|---|---|
| [train_mlp.py](train_mlp.py) | Train an MLP emotion classifier on frozen VLM embeddings. |
| [evaluate_mlp.py](evaluate_mlp.py) | Evaluate a saved MLP checkpoint on extracted VLM embeddings. |
| [run_mosei_tfidf.py](run_mosei_tfidf.py) | Train and evaluate the six-label CMU-MOSEI transcript baseline. |
| [mosei_mlp.py](mosei_mlp.py) | Train a six-output MLP on frozen MOSEI embeddings or evaluate a fixed checkpoint. |

Direct invocation: `python scripts/classification/train_mlp.py --help`.
Shared algorithm modules expose functions rather than a command-line interface.
