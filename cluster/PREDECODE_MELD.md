# Reusable MELD Decord frame cache

Predecode the original MELD train, dev, and test videos on a CPU compute node.
The job uses the Qwen `qwen-vl-utils` Decord path with 6 FPS, 4–64 frames, and
224×224 sampled frames, matching the MELD Qwen2.5-VL-3B video request. It saves
the frame tensors and video metadata; the cache does not contain emotion labels
or text prompts.

From `codes/Emo-benchmark`:

```bash
~/hf-env/bin/python -u cluster/predecode_meld.py --root ../.. --verify-only
~/hf-env/bin/python -u cluster/predecode_meld.py --root ../.. \
  --chunk-size 500 --workers 6 --save-every 25 --decode-timeout-seconds 10
```

Submit the second command in a CPU allocation. With 12 allocated CPUs, start
with six workers. Each worker keeps its own Decord child process; the parent
checkpoints results as clips finish. Changing `--workers` on resume is safe.
Run the same command again after interruption to reuse saved decodes. Failed
clips stay skipped unless `--retry-failed` is specified. `--splits train`,
`--splits dev`, and `--splits test` can limit a run to selected splits; run
these jobs sequentially when they share the same output directory, because
the success and failure manifests are rewritten after each chunk.

The default output is
`../../datasets/meld/decoded_qwen_6fps_224_v1/`. It contains `frames/` with
atomically saved `.pt` and identity `.json` files, resumable `chunks/` status,
`successful_decodes.csv`, `failed_decodes.csv`, `summary.json`, `run_config.json`,
and `run.log`. The success manifest lists `split`, `sample_id`, and `cache_path`.
The summary counts available indexed clips; the known missing annotated MELD
dev video is excluded before decoding. Cache identities include the source
video size and modification time, the `qwen-vl-utils` version, and frame
sampling settings. Use a new output directory if those settings change.

The Qwen2.5-VL-7B MELD full baseline and seven pruning launchers read this
cache. The older MELD face-only Qwen2.5-VL-3B runner still decodes directly.
