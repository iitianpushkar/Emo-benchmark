# LM-head four-condition modality comparison

Run from `codes/Emo-benchmark` on Rorqual after pulling this revision. Both
commands use the Qwen2.5-VL-7B model under `../../models/qwen` and the existing
6 fps, 224 px decoded-frame caches.

```bash
~/hf-env/bin/python -u cluster/run_qwen7b_lm_head_visual_ablation.py \
  --dataset mosei --root ../.. \
  --aligned-csv ../../datasets/mosei/csv/mosei_aligned.csv \
  --decoded-cache-dir ../../datasets/mosei/decoded_qwen_6fps_224_v1

~/hf-env/bin/python -u cluster/run_qwen7b_lm_head_visual_ablation.py \
  --dataset meld --root ../.. \
  --decoded-cache-dir ../../datasets/meld/decoded_qwen_6fps_224_v1
```

Add `--verify-only` to print the selected test counts and emotion-code mapping
without loading the GPU model. Add `--limit 2` for a small development run; use
a different `--output-dir` for that run, then omit `--limit` for the full test.

Only test clips with valid decoded-frame cache entries are selected. The full
pass uses video plus utterance. The no-visual-token pass uses that same video
prompt, then removes projected video tokens at the language-model input. Its
vision encoder still runs. The utterance-only pass constructs a text-only
prompt and does not pass frames or invoke the vision encoder. The video-only
pass uses the same cached frames as the full pass and a prompt with no utterance
text. The same emotion choices and LM-head scoring are used for all four
conditions. Compare full versus utterance-only for the effect of adding video;
compare full versus video-only for the effect of adding the utterance. The
no-visual-token condition also shows the effect of retaining the video prompt
structure and text positions without visual embeddings.

For MELD, A–G map to anger, disgust, fear, joy, neutral, sadness, and surprise.
The LM-head logits for those seven single-token codes are softmax-normalized.
For MOSEI, A–F map to anger, disgust, fear, happy, sad, and surprise. Each is
asked as an independent Y/N question, with positive ground truth defined by a
score greater than zero in the full `mosei_aligned.csv`. MOSEI therefore runs
six question forwards in each condition per clip. Each Y/N pair is normalized
separately; its probabilities are not a seven-class distribution.

Outputs are in `../../results/qwen2_5_vl_7b_{dataset}_lm_head_four_conditions_test/`:

- `selected_test.csv`: exact selected test rows.
- `per_emotion_probabilities.csv`: all four probabilities, pairwise changes,
  ground truth, predictions, and visual-token count for every emotion and clip.
- `summary.json`: aggregate positive ground-truth probability changes by class
  for all six pairs of conditions.
- `lm_head_probability_comparison.png`: class means and change histograms for
  full versus each other condition.
- `run_config.json`: model, protocol, code-token IDs, and selection fingerprint.

The CSV is checkpointed every 10 completed clips and when a clip fails. A
restart resumes whole completed clips after checking the run configuration.
Probability differences use the order named in each column; positive values
mean the first condition increased probability of that emotion. These are candidate-normalized
LM-head probabilities at the first answer token, not calibrated probabilities
over the entire vocabulary.
