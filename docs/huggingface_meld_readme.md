---
language:
- en
task_categories:
- video-classification
- text-classification
tags:
- emotion-recognition
- sentiment-analysis
- multimodal
- conversation
---

# MELD raw videos and annotations

This is a mirror of the locally available MELD 1.0 raw release. MELD
(Multimodal EmotionLines Dataset) contains conversation clips from the television
series *Friends*, with one of seven emotion labels and a three-way sentiment
label per annotated utterance. The creators publish the
[official dataset](https://huggingface.co/datasets/declare-lab/MELD) and
[source repository](https://github.com/declare-lab/MELD). Please cite
[Poria et al., ACL 2019](https://aclanthology.org/P19-1050/) when using MELD.

## Files

- `train/train_sent_emo.csv` and `train/train_splits/*.mp4`
- `dev_sent_emo.csv` and `dev_splits_complete/*.mp4`
- `test_sent_emo.csv` and `output_repeated_splits_test/*.mp4`
- `README.txt`: documentation bundled with the raw release

The CSV files have 9,989 train, 1,109 dev, and 2,610 test rows (13,708 total).
This copy has 13,848 MP4 files. Every annotated train and test clip is present.
The annotated dev clip `dia110_utt7.mp4` is absent from the source release;
there are also 4 dev and 137 test MP4 files without matching annotation rows.
The authors' [issue #22](https://github.com/declare-lab/MELD/issues/22)
documents this source-data discrepancy. Consumers should join CSV rows to
video filenames rather than assume every MP4 has a label.

The upstream Hugging Face repository marks MELD as GPL-3.0. The source footage
comes from *Friends*; the upstream repository does not state that the television
footage itself is GPL-licensed. Consult the creators' release and applicable
rights before further redistribution or reuse.
