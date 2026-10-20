#!/usr/bin/env python3
"""Summarize per-sample Information Horizon outputs without inferring a cutoff."""

# Allow both `python scripts/<group>/<file>.py` and `python -m scripts...`.
import sys as _sys
from pathlib import Path as _Path
if __package__ in (None, ""):
    _sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input-dir', type=Path, required=True)
    a = p.parse_args()
    groups = defaultdict(list)
    for path in sorted(a.input_dir.glob('*.json')):
        if path.name == 'config.json':
            continue
        sample = json.loads(path.read_text())
        y = sample['labels'].index(sample['true_label'])
        full = sample['full_probabilities']
        for d in sample['depths']:
            absent = d['absent_probabilities']
            groups[d['depth']].append(dict(
                mean_information=d['mean_information'],
                mean_absolute_information=d['mean_absolute_information'],
                within_sample_variance=d['variance_information'],
                full_minus_absent_true=full[y]-absent[y],
                full_correct=int(max(range(len(full)), key=full.__getitem__) == y),
                absent_correct=int(max(range(len(absent)), key=absent.__getitem__) == y),
                scored_tokens=len(d['tokens']),
            ))
    if not groups:
        raise ValueError('No completed samples')
    rows = []
    for depth, values in sorted(groups.items()):
        rows.append(dict(depth=depth, samples=len(values), **{k: sum(v[k] for v in values)/len(values) for k in values[0]}))
    target = a.input_dir / 'layer_summary.csv'
    with target.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(target)


if __name__ == '__main__':
    main()
