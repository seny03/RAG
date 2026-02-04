#!/usr/bin/env python3
"""
RAG-based Bug Localization for SWE-bench.

Uses CodeRAG components to find relevant files/functions given an issue description.
"""

import os
import json
import re
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from collections import defaultdict

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


@dataclass
class CodeChunk:
    """A chunk of code from a repository."""
    file_path: str
    start_line: int
    end_line: int
    content: str
    chunk_type: str  # 'function', 'class', 'module'
    name: str


@dataclass
class LocalizationResult:
    """Result of bug localization."""
    file_path: str
    score: float
    matched_chunks: List[CodeChunk]
    

def extract_code_chunks(repo_path: str, max_lines_per_chunk: int = 100) -> List[CodeChunk]:
    """
    Extract code chunks from a Python repository.
    Focuses on functions and classes.
    """
    chunks = []
    
    for root, dirs, files in os.walk(repo_path):
        # Skip hidden and test directories for initial retrieval
        dirs[:] = [d for d in dirs if not d.startswith('.') and d not in ['__pycache__', 'node_modules']]
        
        for file in files:
            if not file.endswith('.py'):
                continue
                
            file_path = os.path.join(root, file)
            rel_path = os.path.relpath(file_path, repo_path)
            
            try:
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()
                    lines = content.split('\n')
            except Exception:
                continue
            
            # Extract functions and classes using simple regex
            # For production, use AST parsing
            current_chunk_start = 0
            current_chunk_name = rel_path
            current_chunk_type = 'module'
            
            for i, line in enumerate(lines):
                # Detect function/class definitions
                func_match = re.match(r'^(async\s+)?def\s+(\w+)', line)
                class_match = re.match(r'^class\s+(\w+)', line)
                
                if func_match or class_match:
                    # Save previous chunk if substantial
                    if i - current_chunk_start > 3:
                        chunks.append(CodeChunk(
                            file_path=rel_path,
                            start_line=current_chunk_start + 1,
                            end_line=i,
                            content='\n'.join(lines[current_chunk_start:i]),
                            chunk_type=current_chunk_type,
                            name=current_chunk_name
                        ))
                    
                    current_chunk_start = i
                    if func_match:
                        current_chunk_name = func_match.group(2)
                        current_chunk_type = 'function'
                    else:
                        current_chunk_name = class_match.group(1)
                        current_chunk_type = 'class'
            
            # Add last chunk
            if len(lines) - current_chunk_start > 3:
                chunks.append(CodeChunk(
                    file_path=rel_path,
                    start_line=current_chunk_start + 1,
                    end_line=len(lines),
                    content='\n'.join(lines[current_chunk_start:]),
                    chunk_type=current_chunk_type,
                    name=current_chunk_name
                ))
    
    return chunks


class SparseRetriever:
    """TF-IDF based sparse retriever for code."""
    
    def __init__(self, chunks: List[CodeChunk]):
        self.chunks = chunks
        self.vectorizer = TfidfVectorizer(
            max_features=10000,
            ngram_range=(1, 2),
            stop_words='english',
            token_pattern=r'(?u)\b[a-zA-Z_][a-zA-Z0-9_]*\b'  # Match identifiers
        )
        
        # Build index
        texts = [f"{c.name} {c.content}" for c in chunks]
        self.tfidf_matrix = self.vectorizer.fit_transform(texts)
        
    def retrieve(self, query: str, top_k: int = 20) -> List[Tuple[CodeChunk, float]]:
        """Retrieve top-k chunks for a query."""
        query_vec = self.vectorizer.transform([query])
        scores = cosine_similarity(query_vec, self.tfidf_matrix).flatten()
        
        top_indices = np.argsort(scores)[::-1][:top_k]
        return [(self.chunks[i], scores[i]) for i in top_indices if scores[i] > 0]


def parse_patch_files(patch: str) -> List[str]:
    """Extract file paths from a git patch."""
    files = set()
    for line in patch.split('\n'):
        if line.startswith('diff --git'):
            # Extract file path from "diff --git a/path/to/file b/path/to/file"
            match = re.search(r'a/(.+?)\s+b/', line)
            if match:
                files.add(match.group(1))
        elif line.startswith('--- a/'):
            files.add(line[6:])
        elif line.startswith('+++ b/'):
            files.add(line[6:])
    return list(files)


def evaluate_localization(
    retrieved_files: List[str],
    gold_files: List[str],
    k_values: List[int] = [1, 5, 10, 20]
) -> Dict[str, float]:
    """
    Evaluate file localization accuracy.
    
    Metrics:
    - Recall@k: How many gold files are in top-k retrieved
    - Precision@k: How many top-k retrieved are gold files
    """
    results = {}
    gold_set = set(gold_files)
    
    for k in k_values:
        retrieved_k = set(retrieved_files[:k])
        
        if len(gold_set) > 0:
            recall = len(retrieved_k & gold_set) / len(gold_set)
        else:
            recall = 0.0
            
        if len(retrieved_k) > 0:
            precision = len(retrieved_k & gold_set) / len(retrieved_k)
        else:
            precision = 0.0
            
        results[f'recall@{k}'] = recall
        results[f'precision@{k}'] = precision
        
        # Hit@k - did we find at least one gold file?
        results[f'hit@{k}'] = 1.0 if len(retrieved_k & gold_set) > 0 else 0.0
    
    return results


def localize_bug(
    issue_text: str,
    repo_path: str,
    retriever: Optional[SparseRetriever] = None,
    top_k: int = 20
) -> List[LocalizationResult]:
    """
    Localize bug to specific files given an issue description.
    
    Args:
        issue_text: The issue/problem statement
        repo_path: Path to the repository
        retriever: Pre-built retriever (optional, will build if None)
        top_k: Number of files to return
        
    Returns:
        List of LocalizationResult with file paths and scores
    """
    if retriever is None:
        print(f"Building index for {repo_path}...")
        chunks = extract_code_chunks(repo_path)
        print(f"Extracted {len(chunks)} chunks")
        retriever = SparseRetriever(chunks)
    
    # Retrieve relevant chunks
    results = retriever.retrieve(issue_text, top_k=top_k * 5)
    
    # Aggregate by file
    file_scores = defaultdict(lambda: {'score': 0.0, 'chunks': []})
    for chunk, score in results:
        file_scores[chunk.file_path]['score'] += score
        file_scores[chunk.file_path]['chunks'].append(chunk)
    
    # Sort by aggregated score
    sorted_files = sorted(file_scores.items(), key=lambda x: x[1]['score'], reverse=True)
    
    return [
        LocalizationResult(
            file_path=file_path,
            score=data['score'],
            matched_chunks=data['chunks'][:3]  # Top 3 chunks per file
        )
        for file_path, data in sorted_files[:top_k]
    ]


def main():
    """Run localization on SWE-bench samples."""
    from datasets import load_dataset
    
    # Load SWE-bench Lite
    print("Loading SWE-bench Lite...")
    ds = load_dataset("princeton-nlp/SWE-bench_Lite", split="test")
    
    # Test on a few samples
    repos_path = "/home/arsemkin/course-project-rag/swebench/repos"
    
    # Map repo names to local paths
    repo_map = {
        'django/django': 'django',
        'sympy/sympy': 'sympy',
        'scikit-learn/scikit-learn': 'scikit-learn',
        'matplotlib/matplotlib': 'matplotlib',
        'pytest-dev/pytest': 'pytest',
        'astropy/astropy': 'astropy',
        'sphinx-doc/sphinx': 'sphinx',
        'pallets/flask': 'flask',
        'psf/requests': 'requests',
        'pydata/xarray': 'xarray',
        'mwaskom/seaborn': 'seaborn',
        'pylint-dev/pylint': 'pylint',
    }
    
    # Build retrievers for each repo
    retrievers = {}
    
    all_results = []
    
    for i, sample in enumerate(ds):
        repo_name = sample['repo']
        if repo_name not in repo_map:
            continue
            
        local_repo = os.path.join(repos_path, repo_map[repo_name])
        if not os.path.exists(local_repo):
            print(f"Skipping {repo_name}: not found at {local_repo}")
            continue
        
        # Build retriever if not cached
        if repo_name not in retrievers:
            print(f"\nBuilding retriever for {repo_name}...")
            chunks = extract_code_chunks(local_repo)
            print(f"  Extracted {len(chunks)} chunks")
            retrievers[repo_name] = SparseRetriever(chunks)
        
        # Get issue and gold files
        issue_text = sample['problem_statement']
        gold_files = parse_patch_files(sample['patch'])
        
        # Localize
        results = localize_bug(
            issue_text,
            local_repo,
            retriever=retrievers[repo_name],
            top_k=20
        )
        
        retrieved_files = [r.file_path for r in results]
        metrics = evaluate_localization(retrieved_files, gold_files)
        
        all_results.append({
            'instance_id': sample['instance_id'],
            'repo': repo_name,
            'gold_files': gold_files,
            'retrieved_files': retrieved_files[:10],
            'metrics': metrics
        })
        
        if i < 5:  # Print first 5 samples
            print(f"\n=== {sample['instance_id']} ===")
            print(f"Issue: {issue_text[:200]}...")
            print(f"Gold files: {gold_files}")
            print(f"Retrieved: {retrieved_files[:5]}")
            print(f"Metrics: {metrics}")
        
        if (i + 1) % 50 == 0:
            print(f"\nProcessed {i + 1} samples...")
    
    # Aggregate metrics
    print("\n" + "="*60)
    print("AGGREGATE RESULTS")
    print("="*60)
    
    metrics_agg = defaultdict(list)
    for r in all_results:
        for k, v in r['metrics'].items():
            metrics_agg[k].append(v)
    
    for k in sorted(metrics_agg.keys()):
        avg = np.mean(metrics_agg[k])
        print(f"{k}: {avg:.3f}")
    
    # Save results
    output_path = "/home/arsemkin/course-project-rag/swebench/localization_results.json"
    with open(output_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
