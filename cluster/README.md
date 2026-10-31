# Cluster staging and preflight

## MELD face-only Qwen2.5-VL-3B experiment

The cluster version of the two Kaggle face-token notebooks is
[`run_meld_qwen2_5vl3b_face_only.py`](run_meld_qwen2_5vl3b_face_only.py).
It uses the same 6 FPS, 4–64 frames, 224 frame-size request, emotion-task
prompt, last-token pooling, RetinaFace detector and 10% patch-overlap rule.
Visual tokens are physically removed at the language-model entry after the
vision encoder. Only the face-only MLP is trained. The runner reuses the tracked
full-video MLP checkpoint and its saved dev/test predictions from
`experiments/baselines/qwen2_5_vl_3b_shared/outputs`. It extracts only the
baseline-covered dev/test clips and reports both methods on the same usable
sample IDs. The saved baseline covers 1,102 dev and 2,599 test clips before
any new face-extraction failures. The dev ablation also applies the saved full-video MLP to
face-only, no-video, time/count-matched random and nonface tokens, and
count-matched DivPrune. No utterance-only MLP is trained.

Put the complete MELD copy at `../../datasets/meld` and download
`Qwen/Qwen2.5-VL-3B-Instruct` into `../../models/qwen2_5_vl_3b` on a login or
transfer node. The older `models/qwen` folder is reserved for 7B; this script
checks the 3B model signature and does not download weights. Load the cluster
OpenCV module before activating the experiment virtual environment. In that
environment, install DeepFace and its RetinaFace detector dependencies, and
stage the detector weights where the compute node can read them. On a networked
login/transfer node, download the ONNX weight file:

```bash
mkdir -p "$HOME/.deepface/weights"
~/hf-env/bin/hf download serengil/deepface retinaface.onnx \
  --local-dir "$HOME/.deepface/weights"
ls -lh "$HOME/.deepface/weights/retinaface.onnx"
```

DeepFace looks in `$DEEPFACE_HOME/.deepface/weights/` when `DEEPFACE_HOME` is
set, otherwise in `$HOME/.deepface/weights/`. If using a project directory,
set `DEEPFACE_HOME` to the same absolute path while staging and in the Slurm
job. The runner's `--verify-only` mode now checks for the weight file.
The tracked baseline files are small enough to arrive with the code checkout;
if they are stored elsewhere, pass `--baseline-dir /path/to/outputs`.
The experiment loads RetinaFace before Qwen and needs no network access on the
compute node once the weights are staged.

From `codes/Emo-benchmark` on the login node:

```bash
~/hf-env/bin/python cluster/run_meld_qwen2_5vl3b_face_only.py --root ../.. --verify-only
```

Then, inside a GPU allocation with the same modules and virtual environment:

```bash
~/hf-env/bin/python -u cluster/run_meld_qwen2_5vl3b_face_only.py --root ../.. --chunk-size 500
```

For a short smoke test, use a **different output directory**:

```bash
~/hf-env/bin/python -u cluster/run_meld_qwen2_5vl3b_face_only.py --root ../.. \
  --limit-per-split 2 --output-dir ../../results/qwen3b_face_smoke
```

The full run processes 500 clips per chunk by default (`--chunk-size N` changes
this). It saves each completed clip immediately under `cache/<split>/` and
atomically writes completed chunks under `chunks/<split>/`. On restart it reuses
complete chunks; if a job stopped within a chunk, it reuses that chunk's saved
clips. Chunks with failures are retried on restart. Once each split finishes,
the runner merges its chunks into `face_<split>.pt`. It then trains one
dev-selected face MLP checkpoint and writes `matched_metrics.csv`,
`matched_predictions.csv`, `paired_comparisons.csv`, `costs.csv`, and
`cost_summary.json`, `coverage.json`, and `failures.csv` beneath
`../../results/qwen2_5_vl_3b_meld_face_only`.
`matched_predictions.csv`, `matched_metrics.csv`, and `costs.csv` put
`original_visual_tokens` beside `retained_visual_tokens`; the metrics and cost
summaries also report both totals and weighted retention. For saved baseline
prediction rows, the original count is reconstructed from this run's matching
video input and retained count equals original because that baseline used no
pruning. The historical baseline did not save per-clip token counts.
Progress appears in the job's stdout and `run.log`; follow it from a login
node with `tail -f ../../results/qwen2_5_vl_3b_meld_face_only/run.log`.
Restart the same command with the same output directory after an interruption.
The runner limits RetinaFace's ONNX thread pool to the CPUs assigned to the
Slurm task and prints its active execution provider. It also prints video
decode, face detection, and Qwen-forward phases for the first clip and every
100th clip, plus progress during face detection whenever a clip takes over
30 seconds. After loading Qwen, it prints the Accelerate device map and
parameter GiB per device, and stops if any weights, forward inputs, or forward
outputs are on CPU. With a CPU-only ONNX provider, increasing `--cpus-per-task` can
improve detector throughput; allocate those CPUs in the Slurm job rather than
letting ONNX create threads across the whole node.
Use `--skip-controls` only for an exploratory run and a separate output folder.
By default, every 100th clip receives a sampled PyTorch registered-operator
TFLOP estimate; change this with `--profile-every N` (or set 0 to disable it).
The cost table flags those profiled forwards because profiler overhead affects
their latency. CUDA peak allocation and exact retained-token counts are also
recorded per clip and condition. The estimator is partial and should be
described as counted-operator FLOPs in a paper. Cost measurements in this run
cover the new face and dev-control forwards; the saved baseline predictions
do not contain comparable new compute-node timing or memory measurements.


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
224 × 224 frames, 6 FPS, and 4–64 frames per clip for the size-matched VLM
comparisons. Each model should receive the same segment transcript and use the
same train/valid/test indexes and six-label classifier procedure. The Qwen2.5
extractor records actual frame dimensions and counts in its metadata.

```bash
python cluster/run_mosei_qwen2_5vl7b_baseline.py --root ../.. --verify-only
```

This checks installed package versions, key imports, and whether PyTorch has
CUDA support, then validates the local model and all MOSEI clip paths. It does
not require a GPU on the login node. The compute-node run separately checks
that CUDA is available.
The baseline and token-curve launchers accept CUDA GPU allocations of any
size, including H100 MIG slices. They print the allocated VRAM before loading
weights; actual memory use depends on the clip and may still exceed a small
slice's capacity.

On Alliance clusters, install missing dependencies from the site wheelhouse
inside the same activated virtual environment. In particular, use a
`torchvision` wheel compatible with the installed Compute Canada PyTorch build;
check the resolver's proposed changes before installing it.

Alliance supplies OpenCV through a module rather than an `opencv-python` pip
wheel. Do not install the generic `requirements.txt` as-is on Alliance; its
`opencv-python` entry triggers the site's dummy wheel. Install other missing
packages separately. Before activating the virtual environment, use
`module spider opencv` to find an available version, then load it with the
appropriate GCC module:

```bash
deactivate
module load gcc
module spider opencv
module load opencv/<available-version>
source /path/to/hf-env/bin/activate
python -c 'import sys; print(sys.executable, sys.prefix)'
python -c 'import cv2; print(cv2.__version__, cv2.__file__)'
```

Load the same modules before activating the environment in every Slurm job.
Check that the printed Python path begins with the virtual environment path;
the `(hf-env)` shell prompt alone does not guarantee that `python` resolves to
the virtual environment. The verifier requires a virtual-environment Python
and checks that required package metadata and imports come from its prefix.
OpenCV is the sole exception because Alliance provides `cv2` via a module.
The verifier checks the actual `cv2` import and reports its version, so the
absence of `opencv-python` pip metadata does not cause a false failure.

Install the updated requirements in the cluster environment before the GPU
job: Qwen2.5 video processing needs `qwen-vl-utils>=0.0.14`,
`torch>=2.5.1` for FLOP counting, and `transformers==4.57.1`, the version used to define the pruning hook's
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
DivPrune hook, and TFLOPs counter before the full sweep:

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

The launcher streams subprocess output to the job's stdout/stderr and appends
the same output to `../../results/qwen2_5_vl_7b_mosei/run.log`. Watch the
terminal in an interactive allocation, or follow the log from a login node:

```bash
tail -F ../../results/qwen2_5_vl_7b_mosei/run.log
```

The extractor reports model loading, each FLOP profile, clip counts, elapsed
time, estimated remaining time, errors, and chunk saves. It reports every
newly attempted clip by default; `--log-every 5` reduces output volume.
Embeddings are checkpointed atomically every 25 newly attempted clips and
at normal chunk completion. Pass `--save-every 5` to reduce the maximum
unsaved progress between checkpoints to four clips. On normal interruption
(Ctrl-C or a scheduler SIGTERM), the extractor also saves completed clips
before exiting. A forced SIGKILL or node failure leaves the last periodic
checkpoint. A restart reuses verified complete chunks and resumes incomplete
chunks from saved sample IDs.

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

The sweep prints each retention ratio and split as well as per-chunk extraction
and MLP training progress to stdout and appends it to
`../../results/qwen2_5_vl_7b_mosei_divprune_curve/run.log`. Follow it with
`tail -F ../../results/qwen2_5_vl_7b_mosei_divprune_curve/run.log`. It uses
the same `--log-every` and `--save-every` options as the baseline.

## 5. Matched random-token control

The [random-token protocol](../experiments/token_pruning/qwen2_5_vl_7b_mosei_random_curve/README.md)
uses the same ratio grid, Qwen checkpoint, video settings, measured costs,
and matched MLP seeds as DivPrune. For paper-level variation estimates, run
three independent extraction seeds in separate H100 jobs:

```bash
python cluster/run_mosei_random_curve.py --root ../.. --verify-only
python cluster/run_mosei_random_curve.py --root ../.. --random-seed 42
python cluster/run_mosei_random_curve.py --root ../.. --random-seed 43
python cluster/run_mosei_random_curve.py --root ../.. --random-seed 44
```

Each seed writes to its own `../../results/qwen2_5_vl_7b_mosei_random_curve_seedN/`
directory and appends live output to `run.log`. Use the comparison command in
the random-token protocol after the DivPrune and random validation curves
finish. Like DivPrune, test evaluation is a separate `--test-all` pass.

After inspecting the full validation curve and fixing the selection rule, run
`python cluster/run_mosei_divprune_curve.py --root ../.. --test-all` to score all
predeclared ratios on test. The curve is computationally large: each ratio
extracts train and validation clips separately, with test clips added by
`--test-all`. It resumes verified chunks after an interrupted job.

All MOSEI experiment launchers record measured per-clip token counts, synchronized extraction
latency, and process GPU allocation. They estimate theoretical forward-pass
TFLOPs with PyTorch `FlopCounterMode` on about 128 evenly spaced clips per
split. This counts registered matrix multiplication, convolution, and attention
formulas and saves the per-operator breakdown. It is a shape-based dense FLOP
estimate, not a measurement of H100 hardware instructions; unsupported and
pointwise operations may be omitted. FLOP counting uses a separate forward and
does not enter latency measurements. See each run's `cost_summary.json` and, for DivPrune,
`validation_curve.csv` and `.png`.

## 6. Deterministic uniform-token control

The [uniform-token protocol](../experiments/token_pruning/qwen2_5_vl_7b_mosei_uniform_curve/README.md)
spreads the same exact video-token budget across Qwen's temporal mRoPE
groups and spatial positions. It shares the DivPrune ratio grid, extraction
settings, three MLP seeds, cost measurements, checkpoints, and live logging.

```bash
python cluster/run_mosei_uniform_curve.py --root ../.. --verify-only
python cluster/run_mosei_uniform_curve.py --root ../..
```

Outputs go to `../../results/qwen2_5_vl_7b_mosei_uniform_curve/`; follow
`run.log` there. The uniform protocol includes a one-clip H100 smoke command
and a comparison command for DivPrune, random, and uniform curves.

## 7. Qwen-adapted VisPruner curve

The [VisPruner protocol](../experiments/token_pruning/qwen2_5_vl_7b_mosei_vispruner_curve/README.md)
uses vision attention and iterative feature deduplication at the same 224 × 224,
6 FPS, 100%–5% retention budgets. Run `--verify-only` on the login node, then
use `python cluster/run_mosei_vispruner_curve.py --root ../..` on a GPU node.
Its resumable progress is visible in
`../../results/qwen2_5_vl_7b_mosei_vispruner_curve/run.log`.

The [VisionZip protocol](../experiments/token_pruning/qwen2_5_vl_7b_mosei_visionzip_curve/README.md)
adapts the authors' Qwen2.5 image code to MOSEI video, retaining dominant
tokens and merging contextual tokens within temporal groups. Run
`python cluster/run_mosei_visionzip_curve.py --root ../.. --verify-only` on the
login node, then remove `--verify-only` in the GPU job. Follow
`../../results/qwen2_5_vl_7b_mosei_visionzip_curve/run.log` for live progress.

## 8. Qwen-adapted FastV curve

The [FastV protocol](../experiments/token_pruning/qwen2_5_vl_7b_mosei_fastv_curve/README.md)
uses the official last-prompt-query attention rule after two decoder layers,
then drops low-attention video tokens before the remaining decoder layers.
It shares the 224 × 224, 6 FPS MOSEI protocol and 100%–5% retention grid.
Run `python cluster/run_mosei_fastv_curve.py --root ../.. --verify-only` on
the login node, then run without `--verify-only` on a GPU node. Follow
`../../results/qwen2_5_vl_7b_mosei_fastv_curve/run.log` for live progress.

## 9. Qwen-adapted SparseVLM curve

The [SparseVLM-style protocol](../experiments/token_pruning/qwen2_5_vl_7b_mosei_sparsevlm_curve/README.md)
uses selected text-token attention and matrix-rank-based deletion counts at
three decoder layers, and recycles
clustered discarded tokens. It uses the same 224 × 224, 6 FPS MOSEI inputs and
100%–5% final visual-token budgets. Run
`python cluster/run_mosei_sparsevlm_curve.py --root ../.. --verify-only` on the
login node, smoke test one clip on the GPU, then run the curve on a GPU node.
Follow `../../results/qwen2_5_vl_7b_mosei_sparsevlm_curve/run.log`.
