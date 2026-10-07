# Consolidated MELD Experiments

This directory compiles the summary metrics in `/Users/pushkarsingh/Downloads/results` without duplicating the multi-gigabyte embedding archives.

## Main results

| Method | Retained input | Test accuracy | Test macro F1 | Test weighted F1 |
| --- | ---: | ---: | ---: | ---: |
| Full video + utterance baseline | 100% | 0.5883 | 0.4336 | 0.5948 |
| DivPrune + new MLP | 70% visual tokens | 0.5795 | 0.4217 | 0.5858 |
| DivPrune + new MLP | 30% visual tokens | **0.6087** | **0.4541** | **0.6127** |
| DINOv2 diverse frames + new MLP | 20% frames | 0.5998 | 0.4417 | 0.6024 |

The strongest observed test result is DivPrune at a 0.30 token-retention ratio. Diverse frame selection at 0.20 also exceeds the recorded baseline while removing 80% of candidate frames before VLM processing.

## Modality findings

On the matched 2,599-sample test set, directly removing visual information reduces macro F1 from 0.4336 to 0.1394. Removing utterance information reduces it to 0.1768. Both modalities therefore contribute strongly under direct ablation.

The conflict experiment gives a different, conditional result: when utterance and video source labels deliberately disagree, predictions follow the utterance label 54.3% of the time and the video label 9.9% of the time; pairwise probability preference favors the utterance label in 80.8% of cases. These source labels are inherited MELD multimodal labels, so this measures model preference rather than independently verified unimodal correctness.

The post-hoc alpha sweep broadly worsens as the video weight increases, with its best macro F1 at zero video weight. Its endpoints should not be equated with the direct-zero ablations: interpolation changes the shared representation distribution, and the fixed classifier was trained on the original joint embeddings.

## Comparability warnings

- The full baseline, both modality ablations, both DivPrune runs, and the alpha sweep use 1,102 dev and 2,599 test samples.
- Frame pruning uses 1,107 dev and 2,609 test samples. Its apparent improvement must be confirmed on the exact shared-ID intersection.
- Results are single runs. Paper-level claims require repeated classifier seeds and uncertainty estimates.
- `divprune_exp_0.3` stores files under a misleading `qwen2_5_vl_3b_divprune_070` path and labels its generated summary as 0.70. Inspection of the serialized `divprune_retain_ratio` and actual retained-token ratios confirms that this archive is the 0.30 run. The consolidated tables correct the label.

## Files

- `main_metrics.csv`: baseline, direct ablations, DivPrune, and frame-pruning results.
- `alpha_sweep.csv`: all video/text interpolation weights.
- `modality_effects.csv`: matched ablation drops and probability-level effects.
- `conflict_summary.csv`: overall conflict preference statistics.
- `experiment_inventory.csv`: archive coverage and provenance notes.
