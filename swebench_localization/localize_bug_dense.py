#!/usr/bin/env python3
"""
RAG-based Bug Localization for SWE-bench with Dense Retrieval.

Uses CodeRAG-style sparse + dense retrieval to find relevant files/functions.
"""

import os
import json
import re
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from collections import defaultdict
import pickle

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import torch
from transformers import AutoTokenizer, AutoModel


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
    """Extract code chunks from a Python repository."""
    chunks = []
    
    for root, dirs, files in os.walk(repo_path):
        dirs[:] = [d for d in dirs if not d.startswith('.') and d not in ['__pycache__', 'node_modules', 'build', 'dist']]
        
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
            
            current_chunk_start = 0
            current_chunk_name = rel_path
            current_chunk_type = 'module'
            
            for i, line in enumerate(lines):
                func_match = re.match(r'^(async\s+)?def\s+(\w+)', line)
                class_match = re.match(r'^class\s+(\w+)', line)
                
                if func_match or class_match:
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


class DenseRetriever:
    """Dense retriever using CodeT5p embeddings."""
    
    def __init__(self, model_name: str = "Salesforce/codet5p-110m-embedding"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Loading model {model_name} on {self.device}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        self.model = AutoModel.from_pretrained(model_name, trust_remote_code=True).to(self.device)
        self.model.eval()
        
        self.chunks = []
        self.embeddings = None
        
    def encode(self, texts: List[str], batch_size: int = 32, max_length: int = 512) -> np.ndarray:
        """Encode texts to embeddings."""
        all_embeddings = []
        
        with torch.no_grad():
            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i:i + batch_size]
                inputs = self.tokenizer(
                    batch_texts,
                    padding=True,
                    truncation=True,
                    max_length=max_length,
                    return_tensors="pt"
                ).to(self.device)
                
                outputs = self.model(**inputs)
                # codet5p-embedding returns embeddings directly as a tensor
                if hasattr(outputs, 'last_hidden_state'):
                    embeddings = outputs.last_hidden_state.mean(dim=1)
                elif isinstance(outputs, torch.Tensor):
                    embeddings = outputs  # Already embeddings
                else:
                    embeddings = outputs[0].mean(dim=1)
                all_embeddings.append(embeddings.cpu().numpy())
        
        return np.vstack(all_embeddings)
    
    def build_index(self, chunks: List[CodeChunk]):
        """Build embedding index for chunks."""
        self.chunks = chunks
        texts = [f"{c.name}\n{c.content[:1000]}" for c in chunks]  # Truncate long chunks
        print(f"Encoding {len(texts)} chunks...")
        self.embeddings = self.encode(texts)
        print(f"Index built with shape {self.embeddings.shape}")
        
    def retrieve(self, query: str, top_k: int = 20) -> List[Tuple[CodeChunk, float]]:
        """Retrieve top-k chunks for a query."""
        query_emb = self.encode([query])[0]
        
        # Cosine similarity
        scores = cosine_similarity([query_emb], self.embeddings).flatten()
        
        top_indices = np.argsort(scores)[::-1][:top_k]
        return [(self.chunks[i], scores[i]) for i in top_indices]


class SparseRetriever:
    """TF-IDF based sparse retriever for code."""
    
    def __init__(self, chunks: List[CodeChunk]):
        self.chunks = chunks
        self.vectorizer = TfidfVectorizer(
            max_features=10000,
            ngram_range=(1, 2),
            stop_words='english',
            token_pattern=r'(?u)\b[a-zA-Z_][a-zA-Z0-9_]*\b'
        )
        
        texts = [f"{c.name} {c.content}" for c in chunks]
        self.tfidf_matrix = self.vectorizer.fit_transform(texts)
        
    def retrieve(self, query: str, top_k: int = 20) -> List[Tuple[CodeChunk, float]]:
        """Retrieve top-k chunks for a query."""
        query_vec = self.vectorizer.transform([query])
        scores = cosine_similarity(query_vec, self.tfidf_matrix).flatten()
        
        top_indices = np.argsort(scores)[::-1][:top_k]
        return [(self.chunks[i], scores[i]) for i in top_indices if scores[i] > 0]


class HybridRetriever:
    """Combines sparse and dense retrieval with score fusion."""
    
    def __init__(self, chunks: List[CodeChunk], dense_model: str = "Salesforce/codet5p-110m-embedding"):
        self.chunks = chunks
        self.sparse = SparseRetriever(chunks)
        self.dense = DenseRetriever(dense_model)
        self.dense.build_index(chunks)
        
    def retrieve(self, query: str, top_k: int = 20, sparse_weight: float = 0.5) -> List[Tuple[CodeChunk, float]]:
        """Retrieve using RRF (Reciprocal Rank Fusion)."""
        k = 60  # RRF constant
        
        # Get sparse results
        sparse_results = self.sparse.retrieve(query, top_k=top_k * 2)
        sparse_ranks = {id(chunk): rank for rank, (chunk, _) in enumerate(sparse_results)}
        
        # Get dense results
        dense_results = self.dense.retrieve(query, top_k=top_k * 2)
        dense_ranks = {id(chunk): rank for rank, (chunk, _) in enumerate(dense_results)}
        
        # RRF fusion
        all_chunks = set()
        chunk_map = {}
        for chunk, _ in sparse_results + dense_results:
            all_chunks.add(id(chunk))
            chunk_map[id(chunk)] = chunk
        
        rrf_scores = {}
        for chunk_id in all_chunks:
            sparse_rank = sparse_ranks.get(chunk_id, 1000)
            dense_rank = dense_ranks.get(chunk_id, 1000)
            
            # Weighted RRF
            sparse_score = sparse_weight / (k + sparse_rank)
            dense_score = (1 - sparse_weight) / (k + dense_rank)
            rrf_scores[chunk_id] = sparse_score + dense_score
        
        # Sort by RRF score
        sorted_chunks = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
        return [(chunk_map[chunk_id], score) for chunk_id, score in sorted_chunks[:top_k]]


def parse_patch_files(patch: str) -> List[str]:
    """Extract file paths from a git patch."""
    files = set()
    for line in patch.split('\n'):
        if line.startswith('diff --git'):
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
    """Evaluate file localization accuracy."""
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
        results[f'hit@{k}'] = 1.0 if len(retrieved_k & gold_set) > 0 else 0.0
    
    return results


def aggregate_to_files(results: List[Tuple[CodeChunk, float]], top_k: int = 20) -> List[str]:
    """Aggregate chunk results to file-level ranking."""
    file_scores = defaultdict(float)
    for chunk, score in results:
        file_scores[chunk.file_path] += score
    
    sorted_files = sorted(file_scores.items(), key=lambda x: x[1], reverse=True)
    return [f for f, _ in sorted_files[:top_k]]


def main():
    """Run localization with hybrid retrieval."""
    from datasets import load_dataset
    
    print("Loading SWE-bench Lite...")
    ds = load_dataset("princeton-nlp/SWE-bench_Lite", split="test")
    
    repos_path = "/home/arsemkin/course-project-rag/swebench/repos"
    cache_path = "/home/arsemkin/course-project-rag/swebench/retriever_cache"
    os.makedirs(cache_path, exist_ok=True)
    
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
    
    # Run evaluation for each retriever type
    for retriever_type in ['sparse', 'dense', 'hybrid']:
        print(f"\n{'='*60}")
        print(f"EVALUATING: {retriever_type.upper()}")
        print('='*60)
        
        retrievers = {}
        all_results = []
        
        for i, sample in enumerate(ds):
            repo_name = sample['repo']
            if repo_name not in repo_map:
                continue
                
            local_repo = os.path.join(repos_path, repo_map[repo_name])
            if not os.path.exists(local_repo):
                continue
            
            # Build retriever if not cached
            if repo_name not in retrievers:
                print(f"\nBuilding {retriever_type} retriever for {repo_name}...")
                chunks = extract_code_chunks(local_repo)
                print(f"  Extracted {len(chunks)} chunks")
                
                if retriever_type == 'sparse':
                    retrievers[repo_name] = SparseRetriever(chunks)
                elif retriever_type == 'dense':
                    retrievers[repo_name] = DenseRetriever()
                    retrievers[repo_name].build_index(chunks)
                else:  # hybrid
                    retrievers[repo_name] = HybridRetriever(chunks)
            
            retriever = retrievers[repo_name]
            
            # Get issue and gold files
            issue_text = sample['problem_statement']
            gold_files = parse_patch_files(sample['patch'])
            
            # Retrieve
            results = retriever.retrieve(issue_text, top_k=40)
            retrieved_files = aggregate_to_files(results, top_k=20)
            
            metrics = evaluate_localization(retrieved_files, gold_files)
            
            all_results.append({
                'instance_id': sample['instance_id'],
                'repo': repo_name,
                'gold_files': gold_files,
                'retrieved_files': retrieved_files[:10],
                'metrics': metrics
            })
            
            if (i + 1) % 100 == 0:
                print(f"  Processed {i + 1} samples...")
        
        # Aggregate metrics
        print(f"\n{retriever_type.upper()} RESULTS:")
        print("-" * 40)
        
        metrics_agg = defaultdict(list)
        for r in all_results:
            for k, v in r['metrics'].items():
                metrics_agg[k].append(v)
        
        for k in sorted(metrics_agg.keys()):
            avg = np.mean(metrics_agg[k])
            print(f"  {k}: {avg:.3f}")
        
        # Save results
        output_path = f"/home/arsemkin/course-project-rag/swebench/localization_{retriever_type}.json"
        with open(output_path, 'w') as f:
            json.dump(all_results, f, indent=2)
        print(f"Results saved to {output_path}")


if __name__ == "__main__":
    main()
