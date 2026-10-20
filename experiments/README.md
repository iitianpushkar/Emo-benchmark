# Experiment catalog

Browse by research question. Each run keeps its notebooks, configuration, and available artifacts together.

| Family | Question |
|---|---|
| [baselines](baselines/README.md) | Full-token Qwen baselines and alternative visual/VLM encoder comparisons. |
| [token_pruning](token_pruning/README.md) | Projected visual-token pruning with DivPrune and random controls. |
| [frame_pruning](frame_pruning/README.md) | Frame selection before the visual encoder, including DINO diversity and random controls. |
| [modality_ablation](modality_ablation/README.md) | Video/text contribution controls and evaluation with fixed classifiers. |
| [conflicts](conflicts/README.md) | Emotion-mismatched utterance/video experiments and their prepared data. |
| [fusion](fusion/README.md) | Weighted combinations of video-only and text-only embeddings. |
| [information_horizon](information_horizon/README.md) | Token-presence/absence interventions across decoder depths. |

## Local inventory

| Run | Available contents |
|---|---|
| [qwen2_5_vl_3b_meld](information_horizon/qwen2_5_vl_3b_meld/README.md) | notebook (automatic full dev Information Horizon; results not yet run) |
| [qwen2_5_vl_3b_shared](baselines/qwen2_5_vl_3b_shared/README.md) | chunks, embeddings, outputs, results |
| [qwen2_5_vl_3b_visual_only](baselines/qwen2_5_vl_3b_visual_only/README.md) | config.yaml |
| [tdv_dino_ibot](baselines/tdv_dino_ibot/README.md) | notebooks |
| [qwen2_5_vl_3b_divprune_030](token_pruning/qwen2_5_vl_3b_divprune_030/README.md) | checkpoints |
| [qwen2_5_vl_3b_dinov2_base_frames_020](frame_pruning/qwen2_5_vl_3b_dinov2_base_frames_020/README.md) | embeddings |
| [random_frames_020](frame_pruning/random_frames_020/README.md) | notebooks |
| [alpha_0_1_frameprune_020_mlp](modality_ablation/alpha_0_1_frameprune_020_mlp/README.md) | outputs |
| [alpha_0_both_mlps](modality_ablation/alpha_0_both_mlps/README.md) | outputs |
| [alpha_1_divprune_030_mlp](modality_ablation/alpha_1_divprune_030_mlp/README.md) | outputs |
| [video_only_divprune_030](modality_ablation/video_only_divprune_030/README.md) | PROTOCOL.md, notebooks |
| [meld_conflict_dataset](conflicts/meld_conflict_dataset/README.md) | data |
| [qwen2_5_vl_3b_conflict_dev](conflicts/qwen2_5_vl_3b_conflict_dev/README.md) | embeddings, outputs |

For cross-experiment metrics see [compiled reports](../reports/compiled_experiments/README.md). Some compared runs exist only in external downloads; this catalog lists local artifacts, not every historical experiment.

## New runs

Use `experiments/<family>/<descriptive_run_name>/`. Keep sampling/pruning settings distinct across runs. Save runtime settings with outputs and use dev data for selection. See [workflows](../docs/workflows.md).
