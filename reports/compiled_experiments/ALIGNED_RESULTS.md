# Sample-Aligned MELD Results

All metrics below were recomputed from prediction CSVs on the same 2,598 test
sample IDs. The global intersection includes the baseline, every pruning run,
both direct modality ablations, and every alpha-sweep prediction file.

## Retrained pruning comparison

| Method | Retained representation | n | Accuracy | Macro F1 | Weighted F1 | Delta macro F1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Full baseline | 100% | 2,598 | 0.5881 | 0.4336 | 0.5946 | 0.0000 |
| DivPrune | 70% tokens | 2,598 | 0.5793 | 0.4216 | 0.5857 | -0.0120 |
| **DivPrune** | **30% tokens** | **2,598** | **0.6085** | **0.4541** | **0.6125** | **+0.0204** |
| DivPrune | 15% tokens | 2,598 | 0.5893 | 0.4417 | 0.5942 | +0.0081 |
| DivPrune | 5% tokens | 2,598 | 0.5689 | 0.4278 | 0.5865 | -0.0059 |
| Random pruning | 30% tokens | 2,598 | 0.5858 | 0.4264 | 0.5900 | -0.0072 |
| Random pruning | 5% tokens | 2,598 | 0.5651 | 0.4291 | 0.5849 | -0.0045 |
| DINOv2 diverse frames | 20% frames | 2,598 | 0.6001 | 0.4394 | 0.6026 | +0.0058 |

## Controlled contrasts

At 30% token retention, DivPrune exceeds random pruning by 0.0227 accuracy,
0.0277 macro F1, and 0.0225 weighted F1. This is the clearest evidence that
diversity selection is useful beyond token reduction alone.

At 5% token retention, DivPrune and random pruning are effectively tied:
DivPrune is higher by 0.0038 accuracy and 0.0016 weighted F1, while random is
higher by 0.0013 macro F1. The selection advantage therefore disappears at
the most aggressive tested token budget.

DINOv2 frame selection at 20% exceeds the full baseline by 0.0119 accuracy,
0.0058 macro F1, and 0.0080 weighted F1. This is a fully matched result, not an
artifact of the frame experiment's larger native sample count.

## Modality ablations

| Method | n | Accuracy | Macro F1 | Weighted F1 |
| --- | ---: | ---: | ---: | ---: |
| Full baseline | 2,598 | 0.5881 | 0.4336 | 0.5946 |
| Zero visual contribution | 2,598 | 0.3857 | 0.1394 | 0.3267 |
| Zero utterance contribution | 2,598 | 0.3584 | 0.1769 | 0.3438 |

Both modalities are important under direct ablation. These results do not
support a claim that visual information is irrelevant, even though visual
frames and tokens are highly redundant.

## Alignment audit

- Baseline and every DivPrune result: 2,599 native IDs, exactly equal ID sets.
- Random 0.30: 2,606 native IDs; all 2,599 baseline IDs plus 7 extras.
- Random 0.05: 2,609 native IDs; all 2,599 baseline IDs plus 10 extras.
- DINOv2 frame 0.20: 2,609 native IDs; 2,598 baseline IDs, 1 baseline ID
  missing, and 11 extras.
- Global intersection used for every table: 2,598 IDs.

The archive named `divprune_exp_0.3` has an incorrect internal `0.70` label.
Its serialized metadata previously confirmed a true retention ratio of 0.30,
so it is reported as 0.30 here.

## Reproducibility files

- `aligned_test_metrics.csv`: all recomputed metrics, including previous MLPs,
  direct ablations, and the complete alpha sweep.
- `aligned_test_deltas.csv`: the same metrics with baseline deltas.
- `sample_alignment_audit.csv`: native counts and exact overlap counts.
- `aligned_test_sample_ids.csv`: the 2,598 IDs used in every calculation.
- `scripts/compile_aligned_results.py`: compiler that reads the downloaded ZIP
  archives directly and recreates these files.
