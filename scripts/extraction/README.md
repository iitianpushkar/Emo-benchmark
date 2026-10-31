# Extraction

Model embedding extractors, chunk runners, and merge utilities.

| Module | Description |
|---|---|
| [extract_qwen_embeddings.py](extract_qwen_embeddings.py) | Extract frozen Qwen2.5-VL video embeddings for MELD. |
| [visualize_qwen_video_tokens.py](visualize_qwen_video_tokens.py) | Overlay projected Qwen2.5-VL token ordinals on their sampled video frames. |
| [extract_qwen_shared_embeddings.py](extract_qwen_shared_embeddings.py) | Extract shared Qwen2.5-VL video+utterance embeddings for MELD. |
| [extract_internvl3_embeddings.py](extract_internvl3_embeddings.py) | Extract frozen InternVL3 shared video-text embeddings for MELD. |
| [extract_smolvlm2_embeddings.py](extract_smolvlm2_embeddings.py) | Extract frozen SmolVLM2 shared video-text embeddings for MELD. |
| [run_qwen_shared_chunks.py](run_qwen_shared_chunks.py) | Run Qwen shared embedding extraction in separate resumable chunks. |
| [run_internvl3_chunks.py](run_internvl3_chunks.py) | Run InternVL3 extraction through the shared resumable chunk runner. |
| [run_smolvlm2_chunks.py](run_smolvlm2_chunks.py) | Run SmolVLM2 embedding extraction in separate resumable processes. |
| [merge_embedding_chunks.py](merge_embedding_chunks.py) | Merge chunked embedding .pt files produced by extraction scripts. |

Direct invocation: `python scripts/extraction/extract_qwen_embeddings.py --help`.
Shared algorithm modules expose functions rather than a command-line interface.

To see the mapping for one video, run this with the same sampling settings as
the Qwen extraction you want to inspect (defaults: 6 FPS, 4–64 frames, size 224):

```bash
python scripts/extraction/visualize_qwen_video_tokens.py /path/to/clip.mp4 \
  --output-dir /path/to/token-map
```

Open `token_map.mp4` to see the global visual token ordinal on every sampled
frame. `token_map.csv` lists each token's sampled-frame pair and rectangle on
the processor-resized frame. Add `--token 42` to highlight only token 42. The
script downloads only the Qwen processor/configuration, not model weights.
