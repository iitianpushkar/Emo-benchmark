# Cluster staging and preflight

This is the first handoff step for the H100 cluster. It does not run an
experiment. It checks access, downloads complete dataset and model snapshots,
then verifies them from a compute node.

The dataset repositories are `0xPushkar/MELD` and `0xPushkar/CMU-MOSEI`.
The model repositories are Qwen2.5-VL-3B-Instruct, InternVL3-2B-hf,
SmolVLM2-2.2B-Instruct, and DINOv2-base, matching the extractor defaults.
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

`--existing-model NAME=PATH` can be repeated for `qwen`, `internvl3`,
`smolvlm2`, and `dinov2`. An explicit model path must contain a config and
complete weight shards; Qwen also needs its tokenizer and processor files.
Missing models are downloaded into `models/`. If staging
fails, no valid manifest is written. Re-running an interrupted Hub download
can reuse Hugging Face's local metadata when the same target directory is used.

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
all four model snapshots, required Python packages, and CUDA visibility. It
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
