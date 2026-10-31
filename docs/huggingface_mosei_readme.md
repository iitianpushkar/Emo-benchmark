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
---

# CMU-MOSEI segment videos and labels

This repository contains a local copy of the available CMU-MOSEI segment videos,
transcripts, emotion scores, sentiment labels, and alignment metadata. The
dataset was introduced by [Zadeh et al. (ACL 2018)](https://aclanthology.org/P18-1208/)
and is described on the [CMU MultiComp dataset page](https://multicomp.cs.cmu.edu/resources/cmu-mosei-dataset/).

## Contents

- `Raw/<video_id>/<clip_id>.mp4`: 22,856 pre-cut segment videos.
- `label.csv`: processed transcript, split, and sentiment annotations.
- `metadata/*.csd`: CMU-MOSEI label and timestamped-word data.
- `csv/mosei_aligned.csv`: 22,856 aligned rows with the train/valid/test split,
  transcript, timestamps, sentiment, and six emotion scores.
- `csv/mosei_labels.csv`, `csv/mosei_segments.csv`, and
  `csv/mosei_timestamped_words.csv`: tabular exports of the CSD metadata.

The aligned rows comprise 16,326 train, 1,871 validation, and 4,659 test
segments. All aligned rows have a corresponding MP4 in this copy. These counts
describe the available files here; the original paper reports 23,453 annotated
segments. A score greater than zero indicates a positive label for each of the
six independent emotions: happy, sad, anger, surprise, disgust, and fear.

The original paper states that the source videos were collected under Creative
Commons terms allowing redistribution. The paper does not identify one license
identifier for every component in this copy; consult the source publication and
upstream dataset for reuse terms. Please cite the original paper when using the
dataset.
