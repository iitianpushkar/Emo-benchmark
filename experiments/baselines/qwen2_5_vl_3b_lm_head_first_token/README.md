# Qwen2.5-VL-3B first-token LM-head baseline

Zero-shot MELD test classification using the frozen Qwen language-model head.
The experiment reuses the cleaned final, last-token shared embeddings from
[`qwen2_5_vl_3b_shared`](../qwen2_5_vl_3b_shared/README.md), so it does not
decode videos or train a classifier.

For each sample, the notebook projects the saved 2048-dimensional final hidden
state through Qwen's original LM head. It records the first generated token for
each of the seven canonical MELD labels, then predicts the label with the
largest corresponding logit. Multi-token labels are intentionally scored only
by their first contextual token in this baseline.

The output distinguishes:

- raw probability under Qwen's complete vocabulary;
- probability renormalized across the seven emotion first tokens;
- unrestricted top vocabulary token;
- total probability mass assigned to the seven candidate tokens.

Run [the Kaggle notebook](notebooks/qwen2_5_vl_3b_lm_head_first_token_test.ipynb)
from top to bottom with the current `benchmark-code` dataset attached. No MELD
video dataset is required because the test embeddings are already saved.

This result is not directly supervision-matched to the trained MLP probe: the
LM-head baseline is zero-shot, while the MLP uses MELD train labels.
