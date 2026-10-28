# Detectorv2 speaker-face prompt on CMU-MOSEI

Run the three Kaggle notebooks independently with the current
`pushkarsingh2005/benchmark-code` input and the
`pushkarsingh2005/cmu-mosei-raw-videos-and-csvs` input:

- [Train](notebooks/detectorv2_speaker_prompt_mosei_train.ipynb): 14,273 included clips, 29 chunks.
- [Dev (source `valid` split)](notebooks/detectorv2_speaker_prompt_mosei_dev.ipynb): 1,647 included clips, 4 chunks.
- [Test](notebooks/detectorv2_speaker_prompt_mosei_test.ipynb): 4,104 included clips, 9 chunks.

Each clip receives one ground-truth class: its unique highest positive emotion
score. All six scores of zero mean `neutral`. A tied highest positive score has
no unique winner, so that clip is excluded. In the current aligned CSV this
excludes 2,053 train, 224 dev, and 555 test clips. The index keeps the source
scores and a `label_status` audit column. Class IDs are happy=0, sad=1,
anger=2, surprise=3, disgust=4, fear=5, neutral=6. This single-label target
is specific to this experiment; the repository's other MOSEI experiments keep
their six-label target.

Each notebook has `START_CHUNK` and `NUM_CHUNKS` settings and writes one `.pt`
file per 500 clips (the last chunk is shorter) beneath
`/kaggle/working/detectorv2_prompt_mosei_singlemax/chunks/<split>/`. Start with one chunk
per session. Completed chunks are skipped; incomplete chunks resume saved
embeddings and retry recorded errors. To continue in a new Kaggle session,
attach the previous notebook output as an input and set `PREVIOUS_CHUNKS_DIR`
to its `chunks/<split>` directory. Save a Kaggle notebook version after each
session so its chunk files persist. The merge cell runs only when all chunks for
that split are present and error free, producing `mosei_<split>.pt`.

The Kaggle video directory is
`/kaggle/input/datasets/pushkarsingh2005/cmu-mosei-raw-videos-and-csvs/Raw/Raw`.
The notebooks locate `mosei_aligned.csv` in the attached MOSEI or code input,
then pass the CSV and video directory separately to `prepare_mosei.py`. Set
`ALIGNED_CSV` in the paths cell if more than one copy is attached. If the
aligned CSV is absent, attach it before running. The directory-only listing
provided earlier did not establish its file location.

The extractor uses the MELD Detectorv2 face tracking and measurement guide,
with a MOSEI prompt that asks Qwen to choose exactly one of the seven classes.
The prompt never contains the ground-truth label. Face measurements are
uncertain cues, and ambiguous speaker selection is marked unavailable. The
prompt excludes Detectorv2's categorical emotion head. Qwen2.5-VL-3B is
frozen, with 6 FPS, 4–64 frames at 224 pixels and last-token pooling. The
extractor saves an embedding conditioned on the prompt; it does not decode a
text answer from Qwen. Each chunk includes scalar class IDs, face diagnostics,
prompts, sample IDs, errors, and extraction configuration.

After gathering all three merged `.pt` files, train the seven-class MLP with
`scripts/classification/train_mlp.py`. Select its checkpoint on dev (`valid`
in the source CSV), then evaluate test once. Do not use `mosei_mlp.py`, which
expects six independent binary labels.

From the repository root, with the three merged `.pt` files in one accessible
directory:

```bash
python scripts/classification/train_mlp.py \
  --train-pt /path/to/mosei_train.pt \
  --dev-pt /path/to/mosei_valid.pt \
  --test-pt /path/to/mosei_test.pt \
  --output-dir /path/to/detectorv2_mosei_mlp
```
