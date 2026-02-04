#!/usr/bin/env python3
"""Compare bug localization results across different retrieval methods."""

import json
import numpy as np
from collections import defaultdict

print('='*70)
print('SWE-bench Bug Localization - RAG Retrieval Comparison')
print('='*70)

methods = ['sparse', 'dense', 'hybrid']
all_metrics = {}

for method in methods:
    try:
        with open(f'/home/arsemkin/course-project-rag/swebench/localization_{method}.json') as f:
            results = json.load(f)
        
        metrics_agg = defaultdict(list)
        for r in results:
            for k, v in r['metrics'].items():
                metrics_agg[k].append(v)
        
        all_metrics[method] = {k: np.mean(v) for k, v in metrics_agg.items()}
        print(f'\n{method.upper()} ({len(results)} samples):')
        for k in sorted(metrics_agg.keys()):
            print(f'  {k}: {np.mean(metrics_agg[k]):.3f}')
    except FileNotFoundError:
        print(f'\n{method.upper()}: File not found')

print('\n')
print('='*70)
print('SUMMARY TABLE')
print('='*70)
header = f"{'Metric':<15} {'Sparse':>10} {'Dense':>10} {'Hybrid':>10} {'Best':>10}"
print(header)
print('-'*55)

for metric in ['hit@1', 'hit@5', 'hit@10', 'hit@20']:
    vals = []
    for m in methods:
        if m in all_metrics and metric in all_metrics[m]:
            vals.append(all_metrics[m][metric])
        else:
            vals.append(0.0)
    best_idx = np.argmax(vals)
    best = methods[best_idx]
    print(f'{metric:<15} {vals[0]:>10.3f} {vals[1]:>10.3f} {vals[2]:>10.3f} {best:>10}')

print('\n')
print('='*70)
print('IMPROVEMENT vs BASELINE (Sparse)')
print('='*70)
if 'sparse' in all_metrics and 'dense' in all_metrics:
    for metric in ['hit@1', 'hit@5', 'hit@10', 'hit@20']:
        sparse_val = all_metrics['sparse'].get(metric, 0)
        dense_val = all_metrics['dense'].get(metric, 0)
        hybrid_val = all_metrics.get('hybrid', {}).get(metric, 0)
        
        dense_imp = (dense_val - sparse_val) / max(sparse_val, 0.001) * 100
        hybrid_imp = (hybrid_val - sparse_val) / max(sparse_val, 0.001) * 100
        
        print(f'{metric}: Dense {dense_imp:+.1f}%, Hybrid {hybrid_imp:+.1f}%')
