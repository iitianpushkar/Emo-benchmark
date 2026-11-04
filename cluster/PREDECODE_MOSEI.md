# Reusable CMU-MOSEI Decord frame cache

Predecode the **original** 22,856 CMU-MOSEI clips once on a CPU compute node.
The job uses the same `qwen-vl-utils` Decord path and 6-FPS, 224×224,
4-to-64-frame protocol as the Qwen2.5-VL-7B baseline and pruning runs. It
stores sampled frame tensors and video metadata. Text prompts and labels are
not cached, so runs with different prompts can use the same frames.

From `codes/Emo-benchmark`, with the same cluster modules and `hf-env` used by
the Qwen jobs:

```bash
~/hf-env/bin/python -u cluster/predecode_mosei.py --root ../.. --verify-only
~/hf-env/bin/python -u cluster/predecode_mosei.py --root ../.. \
  --chunk-size 500 --save-every 25 --decode-timeout-seconds 10
```

Submit the second command as a **CPU compute job**, with no GPU requested.
Run it again after a wall-time interruption; saved decoded clips are reused,
and saved failures are skipped. Use `--retry-failed` only when intentionally
retrying the failed clips. `--splits train`, `--splits valid`, or `--splits test`
can process one split per sequential job. Do not run multiple writers against
the same cache directory simultaneously.

Default output is `../../datasets/mosei/decoded_qwen_6fps_224_v1/`:

- `frames/<hash-prefix>/<sample-id-hash>.pt` and `.json`: one atomically
  published decoded result and source/protocol identity per successful clip.
- `chunks/<split>/<split>_<start>_<end>.json`: resumable status every 25 new
  clips, immediately after a failure, and at chunk/job exit.
- `successful_decodes.csv`: `split`, `sample_id`, `cache_path`.
- `failed_decodes.csv`: `split`, `sample_id`, `error`.
- `summary.json`: expected, successful, failed, and pending counts per split.
- `run.log`: live output mirrored from the scheduler stream.

The success and failure CSVs are refreshed after each chunk. A clip cached just
before a forced shutdown is recovered on the next run even if its status JSON
was not yet written. A Decord timeout is recorded as a failure and does not
stop later clips. Cache validation checks the clip file's size and modification
time, the Qwen video utility version, and the sampling protocol. Use a new
output directory when changing that protocol.

To consume the cache in the 7B MOSEI baseline or any of the seven pruning
curve launchers, add:

```bash
--decoded-cache-dir ../../datasets/mosei/decoded_qwen_6fps_224_v1/frames
```

The extractor reads only valid cached results when this flag is present. A
missing or invalid cache entry becomes a recorded decode failure; it does not
silently run Decord again. Finish predecoding first, then compare
`successful_decodes.csv` and `failed_decodes.csv` against the experiment's
prediction IDs. The cache helps only experiments with the exact same video
sampling protocol; experiments that intentionally change FPS, frame size, or
frame selection need a separate cache or direct decoding.

For latency comparisons, use the same cache setting across methods. Extraction
metadata records whether each clip used `decoded_cache` or direct `decord`;
the one-time CPU preprocessing cost is reported separately from GPU extraction.
