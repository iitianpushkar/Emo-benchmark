#!/usr/bin/env python3
"""Extract the alpha=1 video-only DivPrune condition and evaluate two fixed MLPs."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import shlex
import subprocess
import sys


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--index-dir', type=Path, required=True)
    parser.add_argument('--baseline-checkpoint', type=Path, required=True)
    parser.add_argument('--divprune-checkpoint', type=Path, required=True)
    parser.add_argument('--output-root', type=Path, required=True)
    parser.add_argument('--chunk-size', type=int, default=500)
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    scripts = Path(__file__).resolve().parent
    checkpoints = {'full_baseline_mlp': args.baseline_checkpoint,
                   'divprune_030_mlp': args.divprune_checkpoint}
    required = list(checkpoints.values()) + [args.index_dir / f'meld_{s}_index.csv' for s in ('dev', 'test')]
    for path in required:
        if not path.is_file():
            raise FileNotFoundError(path)

    def run(name: str, *options: object) -> None:
        command = [sys.executable, str(scripts / name), *map(str, options)]
        print(shlex.join(command), flush=True)
        if not args.dry_run:
            subprocess.run(command, check=True)

    rows = []
    for split in ('dev', 'test'):
        chunks = args.output_root / 'chunks' / split
        embeddings = args.output_root / 'embeddings' / f'meld_{split}.pt'
        run('run_qwen_shared_chunks.py',
            '--index-csv', args.index_dir / f'meld_{split}_index.csv',
            '--chunks-dir', chunks, '--chunk-prefix', split,
            '--chunk-size', args.chunk_size, '--skip-existing-complete',
            '--model-id', 'Qwen/Qwen2.5-VL-3B-Instruct',
            '--modality-mode', 'video_only', '--prompt-style', 'emotion_task',
            '--visual-token-pruning', 'divprune', '--divprune-retain-ratio', '0.30',
            '--fps', 6, '--max-frames', 64, '--min-frames', 4, '--frame-size', 224,
            '--video-max-pixels', 3211264, '--pooling', 'last', '--layer', -1,
            '--save-dtype', 'float32')
        run('merge_embedding_chunks.py', '--chunks-dir', chunks, '--output-pt', embeddings)
        if not args.dry_run:
            import torch
            payload = torch.load(embeddings, map_location='cpu', weights_only=False)
            if not torch.isfinite(payload['embeddings']).all():
                raise ValueError(f'Non-finite embeddings in {embeddings}')
            for checkpoint in checkpoints.values():
                saved = torch.load(checkpoint, map_location='cpu', weights_only=False)
                if saved['label_names'] != payload['label_names']:
                    raise ValueError(f'Label order mismatch: {checkpoint}')
            print(f"{split}: {len(payload['labels'])} samples; {len(payload.get('errors', []))} extraction errors")
        for condition, checkpoint in checkpoints.items():
            output = args.output_root / condition
            run('evaluate_mlp.py', '--checkpoint', checkpoint, '--embeddings-pt', embeddings,
                '--output-dir', output, '--split-name', split, '--device', 'cpu')
            if not args.dry_run:
                metrics = json.loads((output / f'{split}_metrics.json').read_text())
                rows.append({'split': split, 'checkpoint': condition,
                             'samples': len(payload['labels']),
                             **{key: metrics[key] for key in ('accuracy', 'macro_f1', 'weighted_f1')}})
    if not args.dry_run:
        with (args.output_root / 'summary.csv').open('w', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        (args.output_root / 'run_manifest.json').write_text(json.dumps({
            'modality_mode': 'video_only', 'alpha_video': 1.0,
            'prompt_style': 'emotion_task', 'divprune_retain_ratio': 0.30,
            'normalized_before_fusion': False, 'training': False,
            'checkpoints': {k: str(v.resolve()) for k, v in checkpoints.items()},
            'index_dir': str(args.index_dir.resolve()),
        }, indent=2) + '\n')


if __name__ == '__main__':
    main()
