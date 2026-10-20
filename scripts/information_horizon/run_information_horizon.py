#!/usr/bin/env python3
"""Information Horizon Eq. 6, adapted to MELD/Qwen2.5-VL.

LM mode uses raw first-answer-token probabilities (not seven-class softmax).
MLP mode uses the fixed checkpoint's seven-class distribution.
"""

# Allow both `python scripts/<group>/<file>.py` and `python -m scripts...`.
import sys as _sys
from pathlib import Path as _Path
if __package__ in (None, ""):
    _sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path

import torch
from scripts.information_horizon.information_horizon import remove_visual_tokens, information_scores


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--index-csv', type=Path, required=True)
    p.add_argument('--output-dir', type=Path, required=True)
    p.add_argument('--readout', choices=['mlp', 'lm'], default='mlp')
    p.add_argument('--checkpoint', type=Path)
    p.add_argument('--model-id', default='Qwen/Qwen2.5-VL-3B-Instruct')
    p.add_argument('--depths', default='0,4,8,12,16,20,24', help='Completed decoder blocks, or all')
    p.add_argument('--limit', type=int, default=1)
    p.add_argument('--start', type=int, default=0)
    p.add_argument('--max-tokens', type=int, default=0, help='0 scores every token; positive values select evenly spaced token ordinals for a pilot')
    p.add_argument('--score-only', action='store_true', help='Save the ground-truth information score per token instead of all seven per-token probability vectors')
    p.add_argument('--fps', type=float, default=6)
    p.add_argument('--min-frames', type=int, default=4)
    p.add_argument('--max-frames', type=int, default=64)
    p.add_argument('--frame-size', type=int, default=224)
    p.add_argument('--pooling', choices=['last', 'mean', 'max'], default='last')
    p.add_argument('--prompt-style', choices=['emotion_task', 'utterance_only'], default='emotion_task')
    p.add_argument('--dtype', choices=['float32', 'float16', 'bfloat16'], default='float16')
    p.add_argument('--resume', action='store_true')
    a = p.parse_args()
    if a.readout == 'mlp' and a.checkpoint is None:
        p.error('--checkpoint is required with --readout mlp')
    if a.limit < 1 or a.start < 0 or a.max_tokens < 0:
        p.error('limit must be positive; start and max-tokens must be nonnegative')
    return a


def main():
    a = parse_args()
    os.environ['VIDEO_MAX_PIXELS'] = str(a.frame_size**2 * a.max_frames)
    from scripts.extraction.extract_qwen_shared_embeddings import make_message, fix_video_kwargs, pool_hidden_states
    from qwen_vl_utils import process_vision_info
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
    from scripts.classification.evaluate_mlp import EmotionMLP, standardize_with_checkpoint, DEFAULT_LABELS
    import transformers

    labels = DEFAULT_LABELS
    classifier = checkpoint = None
    if a.readout == 'mlp':
        checkpoint = torch.load(a.checkpoint, map_location='cpu', weights_only=False)
        labels = checkpoint['label_names']
        if set(labels) != set(DEFAULT_LABELS):
            raise ValueError('Checkpoint must contain the seven MELD emotions')
        classifier = EmotionMLP(checkpoint['input_dim'], checkpoint['hidden_dim'], len(labels), checkpoint.get('dropout', 0))
        classifier.load_state_dict(checkpoint['model_state_dict'])
        classifier.eval()
    config = {k: str(v) if isinstance(v, Path) else v for k, v in vars(a).items() if k not in ('resume', 'output_dir')}
    config.update(implementation='information_horizon_physical_v1', transformers=transformers.__version__, torch=torch.__version__, paper='https://arxiv.org/abs/2512.07580', index_sha256=hashlib.sha256(a.index_csv.read_bytes()).hexdigest())
    if a.checkpoint:
        config['checkpoint_sha256'] = hashlib.sha256(a.checkpoint.read_bytes()).hexdigest()
    a.output_dir.mkdir(parents=True, exist_ok=True)
    manifest = a.output_dir / 'config.json'
    if manifest.exists():
        if not a.resume or json.loads(manifest.read_text()) != config:
            raise ValueError('Output exists; use --resume with identical inputs/settings or a fresh directory')
    else:
        manifest.write_text(json.dumps(config, indent=2))
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(a.model_id, torch_dtype=getattr(torch, a.dtype), device_map='auto', attn_implementation='eager').eval()
    model.requires_grad_(False)
    base = model.model
    language = base.language_model
    depths = list(range(len(language.layers))) if a.depths == 'all' else sorted(set(map(int, a.depths.split(','))))
    if not depths or any(d < 0 or d >= len(language.layers) for d in depths):
        raise ValueError(f'Depths must be 0..{len(language.layers)-1}')
    processor = AutoProcessor.from_pretrained(a.model_id, min_pixels=128*28*28, max_pixels=a.frame_size**2)
    answer_ids = [processor.tokenizer.encode(name, add_special_tokens=False)[0] for name in labels]
    if a.readout == 'lm' and len(set(answer_ids)) != len(labels):
        raise ValueError('Emotion names share first tokens; cannot distinguish them with paper readout')
    with a.index_csv.open() as f:
        rows = list(csv.DictReader(f))[a.start:a.start+a.limit]
    if not rows or len({r['sample_id'] for r in rows}) != len(rows):
        raise ValueError('Empty selection or duplicate sample IDs')
    for row in rows:
        sample_id = row['sample_id']
        key = hashlib.sha256(sample_id.encode()).hexdigest()[:20]
        destination = a.output_dir / f'{key}.json'
        if a.resume and destination.exists():
            continue
        target = labels.index(row['emotion'])
        messages = make_message(row['video_path'], row['utterance'], a.fps, a.min_frames, a.max_frames, a.frame_size, a.prompt_style, 'video_text')
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        images, videos, video_kwargs = process_vision_info(messages, return_video_kwargs=True)
        inputs = processor(text=[text], images=images, videos=videos, padding=True, return_tensors='pt', **fix_video_kwargs(video_kwargs)).to(model.device)
        mask = inputs['attention_mask']
        visual = inputs['input_ids'][0] == model.config.video_token_id
        positions = visual.nonzero().flatten()
        n = len(positions)
        if n == 0:
            raise ValueError(f'No video tokens for {sample_id}')
        tokens = list(range(n)) if not a.max_tokens or a.max_tokens >= n else torch.linspace(0, n-1, a.max_tokens).round().long().tolist()

        def probability(depth=None, token=None):
            from contextlib import nullcontext
            context = nullcontext(None) if depth is None else remove_visual_tokens(language, visual, depth, token)
            with torch.inference_mode(), context as keep:
                output = base(**inputs, use_cache=False, return_dict=True, output_hidden_states=False)
                hidden = output.last_hidden_state
                if a.readout == 'lm':
                    logits = model.lm_head(hidden[:, -1]).float()
                    return logits.softmax(-1)[0, answer_ids].cpu()
                effective_mask = mask if keep is None else mask.index_select(-1, keep.to(mask.device))
                # device_map="auto" may put the final decoder block on a different
                # GPU from input_ids/attention_mask. Boolean indexing needs the
                # mask on the hidden-state device.
                effective_mask = effective_mask.to(hidden.device)
                embedding = pool_hidden_states(hidden, effective_mask, a.pooling).cpu().float().unsqueeze(0)
                embedding = standardize_with_checkpoint(embedding, checkpoint)
                return classifier(embedding).softmax(-1)[0]

        full = probability()
        result = dict(sample_id=sample_id, true_label=row['emotion'], labels=labels, readout=a.readout, answer_first_token_ids=answer_ids if a.readout == 'lm' else None, visual_token_count=n, scored_token_ordinals=tokens, visual_sequence_positions=positions.tolist(), full_probabilities=full.tolist(), depths=[])
        print(f'{sample_id}: {n} visual tokens, {1+len(depths)*(1+len(tokens))} forward passes', flush=True)
        for depth in depths:
            absent = probability(depth)
            entries = []
            for token in tokens:
                present = probability(depth, token)
                delta = information_scores(present, absent)
                entry = dict(token_ordinal=token, sequence_position=int(positions[token]), information_score=float(delta[target]))
                if a.score_only:
                    entry['present_true_probability'] = float(present[target])
                else:
                    entry['probabilities'] = present.tolist()
                    entry['probability_differences'] = delta.tolist()
                entries.append(entry)
            scores = torch.tensor([e['information_score'] for e in entries])
            result['depths'].append(dict(depth=depth, absent_probabilities=absent.tolist(), full_minus_absent=(full-absent).tolist(), mean_information=float(scores.mean()), mean_absolute_information=float(scores.abs().mean()), variance_information=float(scores.var(unbiased=False)), tokens=entries))
            print(f'  depth={depth}: mean I={scores.mean():.6g}; mean |I|={scores.abs().mean():.6g}', flush=True)
        temporary = destination.with_suffix('.tmp')
        temporary.write_text(json.dumps(result, indent=2, allow_nan=False))
        temporary.replace(destination)


if __name__ == '__main__':
    main()
