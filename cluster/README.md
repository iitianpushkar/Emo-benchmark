# Cluster staging and preflight

This is the first handoff step for the H100 cluster. It does not run an
experiment. It checks access, downloads complete dataset and model snapshots,
then verifies them from a compute node.

The dataset repositories are `0xPushkar/MELD` and `0xPushkar/CMU-MOSEI`.
The default preflight selects only Qwen3-VL-8B-Instruct for the current
cluster test. InternVL3-2B-hf, SmolVLM2-2.2B-Instruct, and DINOv2-base remain
available for later runs with `--models all`.
`stage` resolves repositories it downloads to commit SHAs and writes those
revisions to `manifest.json`. It downloads full snapshots, including model
weights and video clips, when complete local copies are unavailable. Use a
shared project or scratch directory with enough space; the
directory must appear at the same absolute path on login and compute nodes.

## 1. On a login or transfer node

Clone the GitHub repository and install the project requirements in the
cluster's Python environment. Ensure `huggingface_hub` is installed. If any
repository is private or gated, configure a **read** token in that environment
through the cluster's approved secret mechanism. Do not put a token in a job
script, Git commit, or log. Acceptance of gated model terms belongs to the
account whose token is used.

From the repository root:

```bash
python -m pip install -r requirements.txt
python cluster/preflight.py probe --report preflight-probe.json
```

`probe` makes no downloads. It checks GitHub and Hugging Face access and lists
whether both dataset repositories contain the required CSVs and video files.
Inspect its JSON report and proceed only if `"ready": true`:

```bash
python cluster/preflight.py stage --root /shared/project/emo-stage \
  --report preflight-stage.json
```

`stage` first checks for a complete MELD or MOSEI copy immediately under the
chosen root's `datasets/` directory. If one exists, it verifies every
annotated clip against the CSVs and reuses that path without downloading it.
For a dataset elsewhere, provide its actual directory:

```bash
python cluster/preflight.py stage --root /shared/project/emo-stage \
  --existing-meld /shared/project/datasets/MELD.Raw \
  --report preflight-stage.json
```

An explicitly supplied path must pass full validation. Existing copies are
recorded as `existing_unpinned` because local files alone do not establish a
Hugging Face commit SHA. Downloaded snapshots are pinned to a SHA. The same
reuse rule applies to a complete model at `models/<name>` beneath the root.
If a model is elsewhere, specify it directly:

```bash
python cluster/preflight.py stage --root /shared/project/emo-stage \
  --existing-model qwen=/project/rrg-basu/tirth/cvpr/models/qwen \
  --report preflight-stage.json
```

`--existing-model NAME=PATH` can be repeated for selected models. An explicit
model path must contain a config and
complete weight shards; Qwen also needs its tokenizer and processor files.
Missing models are downloaded into `models/`. If staging
fails, no valid manifest is written. Re-running an interrupted Hub download
can reuse Hugging Face's local metadata when the same target directory is used.

To stage all model families later, use `--models all` with `probe`, `stage`,
and `check`. To stage a subset, pass names such as `--models qwen dinov2`.

The released MELD copy documented here has one missing annotated dev clip,
`dia110_utt7.mp4`; only that specific gap is accepted. The expected annotation
counts are 9,989/1,109/2,610 for MELD train/dev/test and
16,326/1,871/4,659 for MOSEI train/valid/test.

## 2. Inside a compute-node allocation

Use the same code commit and Python environment. Ask the cluster operator or
your friend to allocate a GPU node with the site's usual scheduler settings;
then run:

```bash
python cluster/preflight.py check --root /shared/project/emo-stage \
  --require-gpu --gpu-name H100 --report preflight-compute.json
```

This checks all annotated video paths again through the compute-node mount,
the selected model snapshot, required Python packages, and CUDA visibility. It
does not load the models or launch an experiment. Exit code 0 and `"ready":
true` mean the staged files and GPU are accessible from that node.

To **also** learn whether the compute node can reach the live services, add
`--network`:

```bash
python cluster/preflight.py check --root /shared/project/emo-stage \
  --require-gpu --gpu-name H100 --network --report preflight-compute-network.json
```

If the first check passes and the second fails, use the staged snapshots for
jobs: compute-node internet access is unnecessary. If either dataset fails,
inspect the report and the staged Hugging Face repository contents before
starting GPU work. The manifest records the exact Hub revisions and the code
commit used while staging.

By default, downloaded paths are `datasets/meld`, `datasets/mosei`, and
`models/{qwen,internvl3,smolvlm2,dinov2}` beneath the chosen root. Existing
dataset paths may differ; use the exact paths in `manifest.json`. Pass the
dataset folders to `prepare_meld --meld-root` and `prepare_mosei --mosei-root`.
Pass the model folders to extraction with `--model-id`; the frame-pruning
encoder uses `--frame-encoder-model`. The future experiment launcher should
read these paths from `manifest.json` rather than re-download model weights on
a GPU node.

Hugging Face documents [revision-pinned snapshot downloads](https://huggingface.co/docs/huggingface_hub/en/guides/download)
and [cache/credential environment variables](https://huggingface.co/docs/huggingface_hub/en/package_reference/environment_variables).

## 3. First experiment: Qwen3-VL-8B on CMU-MOSEI

The first full baseline uses the local Qwen3-VL-8B-Instruct checkpoint and
the six independent MOSEI emotion labels. Check the input paths on the login
node before requesting an H100 job:

The shared [MOSEI video protocol](mosei_video_protocol.json) specifies
448 × 448 frames, 6 FPS, and 4–64 frames per clip for the size-matched VLM
comparisons. Each model should receive the same segment transcript and use the
same train/valid/test indexes and six-label classifier procedure. The Qwen3
extractor records actual frame dimensions and counts in its metadata.

```bash
python cluster/run_mosei_qwen3vl8b_baseline.py --root ../.. --verify-only
```

Install the updated requirements in the cluster environment before the GPU
job: Qwen3 video processing needs `qwen-vl-utils>=0.0.14` and a Transformers
version that provides `Qwen3VLForConditionalGeneration`.

The model config must identify the **Qwen3-VL-8B** checkpoint. An older Qwen
checkpoint in `models/qwen` fails with a clear error. If the 8B checkpoint is
elsewhere, add `--model-dir /actual/path/to/Qwen3-VL-8B-Instruct`.
If `models/qwen` holds the earlier checkpoint, stage the new one in its own
directory on the login node:

```bash
hf download Qwen/Qwen3-VL-8B-Instruct --local-dir ../../models/qwen3_vl_8b
python cluster/preflight.py stage --root ../.. \
  --existing-model qwen=../../models/qwen3_vl_8b \
  --report preflight-stage.json
python cluster/run_mosei_qwen3vl8b_baseline.py --root ../.. \
  --model-dir ../../models/qwen3_vl_8b --verify-only
```

Pass the same `--model-dir ../../models/qwen3_vl_8b` to the GPU command below.

Inside the allocated H100 job, run one command:

```bash
python cluster/run_mosei_qwen3vl8b_baseline.py --root ../..
```

This prepares train/valid/test indexes, extracts frozen video-and-transcript
embeddings for all 22,856 clips, checks complete coverage, trains the six-output
MLP with validation-based checkpoint and threshold selection, and evaluates
the test split. It saves artifacts under `../../results/qwen3_vl_8b_mosei/`.
The launcher uses local model files and sets offline Hub mode for child
processes. Re-running the same command resumes extraction chunks. See the
[baseline protocol](../experiments/baselines/qwen3_vl_8b_mosei/README.md).
