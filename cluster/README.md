# Cluster staging and preflight

This is the first handoff step for the H100 cluster. It does not run an
experiment. It checks access, downloads complete dataset and model snapshots,
then verifies them from a compute node.

The dataset repositories are `0xPushkar/MELD` and `0xPushkar/CMU-MOSEI`.
The default preflight selects only Qwen2.5-VL-7B-Instruct for the current
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

## 3. First experiment: Qwen2.5-VL-7B on CMU-MOSEI

The first full baseline uses the local Qwen2.5-VL-7B-Instruct checkpoint and
the six independent MOSEI emotion labels. Check the input paths on the login
node before requesting an H100 job:

The shared [MOSEI video protocol](mosei_video_protocol.json) specifies
448 × 448 frames, 6 FPS, and 4–64 frames per clip for the size-matched VLM
comparisons. Each model should receive the same segment transcript and use the
same train/valid/test indexes and six-label classifier procedure. The Qwen2.5
extractor records actual frame dimensions and counts in its metadata.

```bash
python cluster/run_mosei_qwen2_5vl7b_baseline.py --root ../.. --verify-only
```

Install the updated requirements in the cluster environment before the GPU
job: Qwen2.5 video processing needs `qwen-vl-utils>=0.0.14` and
`transformers==4.57.1`, the version used to define the pruning hook's
language-model boundary.

The model config must identify the **Qwen2.5-VL-7B** checkpoint. A Qwen3
checkpoint in `models/qwen` fails with a clear error. If the 7B checkpoint is
elsewhere, add `--model-dir /actual/path/to/Qwen2.5-VL-7B-Instruct`.
If `models/qwen` holds another checkpoint, stage the 7B one in its own
directory on the login node:

```bash
hf download Qwen/Qwen2.5-VL-7B-Instruct --local-dir ../../models/qwen2_5_vl_7b
python cluster/preflight.py stage --root ../.. \
  --existing-model qwen=../../models/qwen2_5_vl_7b \
  --report preflight-stage.json
python cluster/run_mosei_qwen2_5vl7b_baseline.py --root ../.. \
  --model-dir ../../models/qwen2_5_vl_7b --verify-only
```

Pass the same `--model-dir ../../models/qwen2_5_vl_7b` to the GPU command below.
On the allocated H100, a one-clip smoke run can validate the model forward,
DivPrune hook, and TFLOPs profiler before the full sweep:

```bash
python scripts/data/prepare_mosei.py --mosei-root ../../datasets/mosei \
  --output-dir ../../results/qwen2_5_smoke/indexes
python cluster/extract_mosei_qwen2_5vl.py \
  --index-csv ../../results/qwen2_5_smoke/indexes/mosei_valid_index.csv \
  --output-pt ../../results/qwen2_5_smoke/valid_30pct.pt \
  --model-dir ../../models/qwen2_5_vl_7b --start 0 --limit 1 \
  --retain-ratio 0.3 --profile-stride 1
```

Inside the allocated H100 job, run one command:

```bash
python cluster/run_mosei_qwen2_5vl7b_baseline.py --root ../..
```

This prepares train/valid/test indexes, extracts frozen video-and-transcript
embeddings for all 22,856 clips, checks complete coverage, trains the six-output
MLP with validation-based checkpoint and threshold selection, and evaluates
the test split. It saves artifacts under `../../results/qwen2_5_vl_7b_mosei/`.
The launcher uses local model files and sets offline Hub mode for child
processes. Re-running the same command resumes extraction chunks. See the
[baseline protocol](../experiments/baselines/qwen2_5_vl_7b_mosei/README.md).

## 4. DivPrune token-retention curve

The [DivPrune protocol](../experiments/token_pruning/qwen2_5_vl_7b_mosei_divprune_curve/README.md)
uses the same checkpoint, MOSEI splits, frame settings, and classifier family.
It sweeps 100% through 5% of projected video tokens with three matched MLP
seeds, recording actual token retention and extraction time. Validate inputs on
the login node, then submit the GPU command inside an H100 allocation:

```bash
python cluster/run_mosei_divprune_curve.py --root ../.. --verify-only
python cluster/run_mosei_divprune_curve.py --root ../..
```

After inspecting the full validation curve and fixing the selection rule, run
`python cluster/run_mosei_divprune_curve.py --root ../.. --test-all` to score all
predeclared ratios on test. The curve is computationally large: each ratio
extracts train and validation clips separately, with test clips added by
`--test-all`. It resumes verified chunks after an interrupted job.

Both launchers record measured per-clip token counts, synchronized extraction
latency, and process GPU allocation, plus TFLOPs per profiled clip for the
operators supported by PyTorch's profiler. The profiler figure is a partial
FLOP estimate, not total model FLOPs. Each split gets about five evenly spaced
profiled clips; profiling uses a separate forward and does not enter latency
measurements. See each run's `cost_summary.json` and, for DivPrune,
`validation_curve.csv` and `.png`.
