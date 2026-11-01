# Qwen2.5-VL-3B MOSEI last-query attention

This experiment visualizes how the final input token attends to CMU-MOSEI
video and prompt tokens at every Qwen language-decoder layer. The final token
is the assistant-generation marker produced by the checkpoint chat template
with `add_generation_prompt=True`.

For every layer, the implementation reconstructs post-RoPE causal attention
for only the final query and computes the arithmetic mean across all attention
heads for each key token. It does not request or retain full
sequence-by-sequence attention matrices.

## Outputs per clip

- `attention_arrays.npz`: raw `[layers, sequence]` head-mean attention and the
  reshaped video-token tensor.
- `layer_summary.csv`: true attention mass before, within, and after the video
  token span, plus final-token self-attention and entropy.
- `visual_token_attention.csv`: one raw probability for every layer and
  spatiotemporal video token.
- `temporal_attention_by_layer.png`: within-video temporal patterns.
- `spatial_attention_all_layers.png`: time-summed spatial maps for every layer.
- `layer_overlays/layer_XX.png`: video-frame contact sheets for every layer.

The CSV and NPZ values remain raw probabilities. Frame-overlay colors use one
linear maximum shared by every layer of a clip, and each overlay header reports
the raw video-attention mass and token-score range. Temporal and spatial summary
heatmaps are normalized within each layer to reveal their internal pattern.
Attention heatmaps are diagnostic evidence, not proof that a token caused the
model output.

Run the Kaggle notebook in
[`notebooks/qwen2_5_vl_3b_mosei_last_query_attention.ipynb`](notebooks/qwen2_5_vl_3b_mosei_last_query_attention.ipynb).
