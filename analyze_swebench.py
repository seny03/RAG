#!/usr/bin/env python3
"""Analyze SWE-bench Lite dataset structure."""

from datasets import load_dataset
import json
import os

print("=== SWE-bench Lite Full Analysis ===")
ds = load_dataset("princeton-nlp/SWE-bench_Lite", split="test")

# Unique repos
repos = list(set(ds["repo"]))
print(f"Unique repos: {len(repos)}")
repo_counts = {}
for sample in ds:
    repo = sample["repo"]
    repo_counts[repo] = repo_counts.get(repo, 0) + 1

for r in sorted(repo_counts.keys()):
    print(f"  {r}: {repo_counts[r]} samples")

print()
print("=== Sample with full problem_statement ===")
sample = ds[0]
print(f"Repo: {sample['repo']}")
print(f"Instance ID: {sample['instance_id']}")
print(f"Base commit: {sample['base_commit']}")
print()
print("Problem Statement:")
print(sample["problem_statement"][:2000])
print()
print("=== Patch (solution) ===")
print(sample["patch"][:1000])

print()
print("=== Comparison with CCEval ===")
print("""
| Aspect | CCEval | SWE-bench Lite |
|--------|--------|----------------|
| Task | Code completion | Bug fixing |
| Input | Incomplete code + context | Issue description |
| Output | Next line(s) of code | Git patch |
| Evaluation | Edit Similarity, Exact Match | Pass@k (tests pass) |
| Repos | 471 (Python only for us) | 12 major OSS projects |
| Samples | ~2600 | 300 |
""")

# Save dataset to parquet for easier access
output_dir = "/home/arsemkin/course-project-rag/swebench"
os.makedirs(output_dir, exist_ok=True)
ds.to_parquet(f"{output_dir}/swebench_lite_test.parquet")
print(f"\nDataset saved to {output_dir}/swebench_lite_test.parquet")

# Also save as JSON for inspection
with open(f"{output_dir}/sample.json", "w") as f:
    json.dump({k: v[:500] if isinstance(v, str) else v for k, v in sample.items()}, f, indent=2)
print(f"Sample saved to {output_dir}/sample.json")
