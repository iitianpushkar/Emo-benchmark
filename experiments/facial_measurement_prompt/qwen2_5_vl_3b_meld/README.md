# Detectorv2 speaker-face prompt on MELD

Run the three split notebooks as independent Kaggle jobs:

- [Train extraction](notebooks/detectorv2_speaker_prompt_meld_train.ipynb) → `meld_train.pt`
- [Dev extraction](notebooks/detectorv2_speaker_prompt_meld_dev.ipynb) → `meld_dev.pt`
- [Test extraction](notebooks/detectorv2_speaker_prompt_meld_test.ipynb) → `meld_test.pt`

Attach the current `pushkarsingh2005/benchmark-code` and
`pushkarsingh2005/meld-dataset` Kaggle inputs and enable Internet and GPU for
each job. Refresh `benchmark-code` from this repository first: the notebooks
run its `scripts/extraction/extract_meld_detectorv2_prompt.py` directly.
Each job extracts only its assigned split and saves its own `.pt` artifact under
`/kaggle/working/detectorv2_prompt/`. Training and classification can be done
later, after all three artifacts are available together.

The install cell removes Kaggle's unused Gradio and Diffusers preinstalls,
which require `huggingface-hub` 1.x. The experiment pins Transformers 4.57.1,
which requires `huggingface-hub` below 1. The next cell checks that Detectorv2,
Qwen, and the video utility import successfully before extraction begins.

The experiment retains the existing Qwen2.5-VL-3B-Instruct protocol: 6 fps,
4–64 frames at 224 pixels, full video plus utterance plus seven-class emotion
task, frozen final shared hidden state, last-token pooling, and a newly trained
MLP chosen on dev. Detectorv2 sees the exact frames decoded for Qwen.

The added prompt describes AU probabilities and blendshape coefficients on
their package-defined [0,1] scale, and gives means, observed ranges, and a
per-frame mouth/jaw timeline. It does not pass Detectorv2's categorical emotion
head to Qwen. Faces are linked by adjacent-frame bounding-box overlap; the
longer, more mouth-active track is marked as the **likely** speaker. If the
selection is weak or ambiguous, the prompt says measurements are unavailable.
Lip movement alone cannot verify the speaker, particularly for silent reaction
shots, camera cuts, occlusion, or dubbing. Inspect the saved `speaker_face`
metadata and the status counts before interpreting gains.

The split notebooks save resumable embeddings, errors, prompts, and face
diagnostics. The earlier [combined notebook](notebooks/detectorv2_speaker_prompt_meld.ipynb)
also contains a smoke test, MLP training, and paired baseline comparison for
use after extraction.

Sources: [Py-Feat Detectorv2 API](https://py-feat.org/api/feat/detector_v2/),
[Py-Feat model card](https://github.com/cosanlab/py-feat/blob/main/feat/multitask/MODEL_CARD.md),
[Py-Feat video tutorial](https://py-feat.org/basic_tutorials/Detecting_Videos/).
