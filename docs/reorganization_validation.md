# Reorganization validation

- All 22 unit/integration tests pass, including a real tiny-Qwen decoder test.
- All 120 recorded hashes of relocated tensors, checkpoints, CSV/JSON artifacts,
  documents, database and log files match their original contents.
- All Python script files parse, registered command targets exist, package imports
  resolve, active notebook script references exist, and catalog links resolve.
- The relocated baseline MLP evaluated on 1,102 cleaned dev embeddings reproduces
  the original sample IDs and predicted classes: accuracy 0.6107, macro F1 0.4872.
  Validation output is temporary and does not overwrite the saved experiment.
- Direct CLI and dispatcher checks passed for tools whose dependencies are installed.
  Extraction/data CLI smoke checks stopped at missing pandas in the local virtual
  environment; model extraction was not run. Install requirements.txt in the
  extraction environment before running GPU experiments.
- Fixed an existing random-pruning test typo: six retained tokens must have six
  unique indices, not three. The pruning algorithm itself was not changed.

Historical metadata and external archive member paths remain unchanged. External
jobs using the former flat script paths need the migration map or command dispatcher.
