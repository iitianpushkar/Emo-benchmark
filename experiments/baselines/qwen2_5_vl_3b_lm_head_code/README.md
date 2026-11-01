# Qwen2.5-VL-3B A-G LM-head baseline

Zero-shot MELD test classification using seven single-token code verbalizers:

```text
A = anger
B = disgust
C = fear
D = joy
E = neutral
F = sadness
G = surprise
```

Unlike the naive natural-label first-token baseline, this experiment
re-extracts the final Qwen hidden state with a prompt ending in `Answer with
only one code:`. It therefore measures the A-G token logits at the generation
position where the model was explicitly instructed to emit a code. Qwen stays
frozen and no classifier is trained.

The [Kaggle notebook](notebooks/qwen2_5_vl_3b_lm_head_code_test.ipynb):

1. rebuilds the MELD test index with Kaggle paths;
2. extracts resumable 6 FPS, 224 x 224 video+utterance embeddings in chunks;
3. aligns successful rows to the cleaned 2,599-sample supervised baseline;
4. verifies A-G are distinct single contextual tokens;
5. applies Qwen's original frozen LM head;
6. saves raw-vocabulary and seven-code probabilities, metrics, and an archive.

This is a test-only, fixed-prompt zero-shot evaluation. No test labels are used
to select or calibrate predictions. The trained MLP remains a
supervision-different reference because it learns from MELD train labels.
