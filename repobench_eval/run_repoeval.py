#!/usr/bin/env python3
"""
Run code completion evaluation on RepoEval dataset.
Uses the pre-computed retrieval from AnhMinhLe/repoeval_comparison_dataset.
"""

import json
import os
from datasets import load_dataset
from collections import defaultdict
import numpy as np
from typing import List, Dict
import time

# Import metrics from CodeRAG
import sys
sys.path.insert(0, '/home/arsemkin/course-project-rag/CodeRAG')
from coderag.evaluation.metrics import compute_ES, compute_EM


def truncate_to_first_line(text: str) -> str:
    """Truncate to first non-empty line."""
    for line in text.split('\n'):
        stripped = line.strip()
        if stripped:
            return stripped
    return text.strip()


def evaluate_completion(pred: str, gold: str) -> Dict[str, float]:
    """Evaluate a single completion."""
    return {
        'edit_sim': compute_ES(gold, pred, language="python"),
        'exact_match': compute_EM(gold, pred, language="python"),
    }


def run_inference_openai(prompts: List[str], model: str, api_url: str, api_key: str, max_tokens: int = 48) -> List[str]:
    """Run inference using OpenAI-compatible API (vLLM)."""
    from openai import OpenAI
    
    client = OpenAI(base_url=api_url, api_key=api_key)
    results = []
    
    for i, prompt in enumerate(prompts):
        try:
            response = client.completions.create(
                model=model,
                prompt=prompt,
                max_tokens=max_tokens,
                temperature=0.0,
                stop=['\n\n']
            )
            results.append(response.choices[0].text)
        except Exception as e:
            print(f"Error on sample {i}: {e}")
            results.append("")
        
        if (i + 1) % 50 == 0:
            print(f"Processed {i + 1}/{len(prompts)} samples...")
    
    return results


def main():
    print("="*70)
    print("RepoEval Code Completion Evaluation")
    print("="*70)
    
    # Load dataset
    print("\nLoading RepoEval dataset...")
    ds = load_dataset('AnhMinhLe/repoeval_comparison_dataset', split='repoeval_bm25_final')
    print(f"Loaded {len(ds)} samples")
    
    # Extract prompts and ground truth
    prompts = []
    ground_truths = []
    metadata_list = []
    
    for sample in ds:
        prompt = sample['prompt']
        metadata = sample['metadata']
        
        # Ground truth is in metadata
        ground_truth = metadata['ground_truth']
        
        prompts.append(prompt)
        ground_truths.append(ground_truth)
        metadata_list.append(metadata)
    
    print(f"Extracted {len(prompts)} prompts")
    
    # Check if vLLM server is available
    api_url = "http://localhost:8001/v1"
    api_key = "sk-xxx"
    model = "Qwen/Qwen2.5-Coder-7B-Instruct"
    
    print(f"\nRunning inference with {model}...")
    print(f"API URL: {api_url}")
    
    try:
        # Test connection
        from openai import OpenAI
        client = OpenAI(base_url=api_url, api_key=api_key)
        test_response = client.models.list()
        print(f"Available models: {[m.id for m in test_response.data]}")
    except Exception as e:
        print(f"ERROR: Cannot connect to vLLM server: {e}")
        print("\nTo start vLLM server, run:")
        print("  vllm serve Qwen/Qwen2.5-Coder-7B-Instruct --port 8001")
        return
    
    # Run inference
    start_time = time.time()
    predictions = run_inference_openai(prompts, model, api_url, api_key)
    inference_time = time.time() - start_time
    print(f"Inference completed in {inference_time:.1f}s ({len(prompts)/inference_time:.1f} samples/sec)")
    
    # Evaluate
    print("\nEvaluating...")
    all_metrics = []
    
    for pred, gold in zip(predictions, ground_truths):
        metrics = evaluate_completion(pred, gold)
        all_metrics.append(metrics)
    
    # Aggregate metrics
    print("\n" + "="*70)
    print("RESULTS")
    print("="*70)
    
    avg_edit_sim = np.mean([m['edit_sim'] for m in all_metrics])
    avg_exact_match = np.mean([m['exact_match'] for m in all_metrics])
    
    print(f"Edit Similarity: {avg_edit_sim:.3f}")
    print(f"Exact Match: {avg_exact_match:.3f}")
    print(f"Samples: {len(all_metrics)}")
    
    # Save results
    results = {
        'model': model,
        'dataset': 'AnhMinhLe/repoeval_comparison_dataset',
        'split': 'repoeval_bm25_final',
        'metrics': {
            'edit_similarity': avg_edit_sim,
            'exact_match': avg_exact_match,
        },
        'num_samples': len(all_metrics),
        'inference_time_sec': inference_time,
    }
    
    output_path = "/home/arsemkin/course-project-rag/repoeval_results.json"
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {output_path}")
    
    # Show some examples
    print("\n" + "="*70)
    print("SAMPLE PREDICTIONS")
    print("="*70)
    for i in range(min(3, len(predictions))):
        print(f"\nSample {i}:")
        print(f"  Function: {metadata_list[i]['function_name']}")
        print(f"  Gold (first line): {truncate_to_first_line(ground_truths[i])[:80]}")
        print(f"  Pred (first line): {truncate_to_first_line(predictions[i])[:80]}")
        print(f"  Edit Sim: {all_metrics[i]['edit_sim']:.3f}, EM: {all_metrics[i]['exact_match']}")


if __name__ == "__main__":
    main()
