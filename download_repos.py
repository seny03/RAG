#!/usr/bin/env python3
"""
Script to download GitHub repositories needed for CCEval benchmark.
Run on the server: python download_repos.py
"""

import json
import os
import subprocess
from pathlib import Path
from typing import Set
import argparse

def extract_repo_info(repo_name: str) -> tuple:
    """
    Extract GitHub username and repo from CCEval format.
    Format: 'username-reponame-commithash'
    """
    parts = repo_name.rsplit('-', 1)
    if len(parts) == 2:
        name_parts = parts[0].rsplit('-', 1)
        if len(name_parts) == 2:
            return name_parts[0], name_parts[1], parts[1]
    # Fallback - try to split differently
    parts = repo_name.split('-')
    if len(parts) >= 3:
        username = parts[0]
        commit = parts[-1]
        repo = '-'.join(parts[1:-1])
        return username, repo, commit
    return None, None, None

def get_repos_from_benchmark(jsonl_path: str) -> Set[str]:
    """Extract unique repository names from benchmark file."""
    repos = set()
    with open(jsonl_path, 'r') as f:
        for line in f:
            data = json.loads(line)
            repos.add(data['metadata']['repository'])
    return repos

def clone_repo(username: str, repo: str, commit: str, output_dir: Path, repo_name: str) -> bool:
    """Clone a repository at specific commit."""
    repo_path = output_dir / repo_name
    
    if repo_path.exists():
        print(f"  [SKIP] {repo_name} already exists")
        return True
    
    # Try multiple possible GitHub URLs
    urls = [
        f"https://github.com/{username}/{repo}.git",
        f"https://github.com/{username}/{repo.replace('-', '_')}.git",
    ]
    
    for url in urls:
        try:
            # Clone
            result = subprocess.run(
                ["git", "clone", "--depth", "1", url, str(repo_path)],
                capture_output=True,
                text=True,
                timeout=120
            )
            
            if result.returncode == 0:
                # Try to checkout specific commit (might fail for shallow clone)
                subprocess.run(
                    ["git", "-C", str(repo_path), "fetch", "--depth", "1", "origin", commit],
                    capture_output=True,
                    timeout=60
                )
                subprocess.run(
                    ["git", "-C", str(repo_path), "checkout", commit],
                    capture_output=True,
                    timeout=30
                )
                print(f"  [OK] {repo_name}")
                return True
        except subprocess.TimeoutExpired:
            print(f"  [TIMEOUT] {repo_name}")
        except Exception as e:
            pass
    
    print(f"  [FAIL] {repo_name} - could not clone")
    return False

def main():
    parser = argparse.ArgumentParser(description='Download CCEval repositories')
    parser.add_argument('--benchmark', type=str,
                        default='/home/arsemkin/course-project-rag/cceval/data/python/line_completion.jsonl',
                        help='Path to benchmark JSONL file')
    parser.add_argument('--output', type=str,
                        default='/home/arsemkin/course-project-rag/CodeRAG/cache/benchmark/cceval/raw_data',
                        help='Output directory for repositories')
    parser.add_argument('--limit', type=int, default=None,
                        help='Limit number of repos to download (for testing)')
    args = parser.parse_args()
    
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Reading benchmark from: {args.benchmark}")
    repos = get_repos_from_benchmark(args.benchmark)
    print(f"Found {len(repos)} unique repositories")
    
    if args.limit:
        repos = list(repos)[:args.limit]
        print(f"Limiting to {args.limit} repositories")
    
    success = 0
    failed = 0
    
    for i, repo_name in enumerate(repos):
        print(f"\n[{i+1}/{len(repos)}] Processing {repo_name}...")
        username, repo, commit = extract_repo_info(repo_name)
        
        if username and repo:
            if clone_repo(username, repo, commit, output_dir, repo_name):
                success += 1
            else:
                failed += 1
        else:
            print(f"  [FAIL] Could not parse repo name: {repo_name}")
            failed += 1
    
    print(f"\n{'='*50}")
    print(f"Download complete!")
    print(f"Success: {success}")
    print(f"Failed: {failed}")
    print(f"Repos saved to: {output_dir}")

if __name__ == "__main__":
    main()
