#!/usr/bin/env python3
"""
Build dataflow graphs for SWE-bench repositories using CodeRAG's static analysis.
"""

import os
import sys
import json
import pickle
from pathlib import Path
from collections import defaultdict

# Add CodeRAG to path
sys.path.insert(0, '/home/arsemkin/course-project-rag/CodeRAG')

from coderag.static_analysis.data_flow.extract_dataflow import PythonParser


def build_graph_for_repo(repo_path: str, output_path: str) -> dict:
    """Build dataflow graph for all Python files in a repository."""
    print(f"Building graph for {repo_path}...")
    
    parser = PythonParser()
    all_graphs = {}
    total_nodes = 0
    total_edges = 0
    files_parsed = 0
    errors = 0
    
    for root, dirs, files in os.walk(repo_path):
        # Skip hidden and build directories
        dirs[:] = [d for d in dirs if not d.startswith('.') and d not in ['__pycache__', 'build', 'dist', 'node_modules', '.git']]
        
        for file in files:
            if not file.endswith('.py'):
                continue
                
            file_path = os.path.join(root, file)
            rel_path = os.path.relpath(file_path, repo_path)
            
            try:
                parser.parse_file(file_path)
                
                # Get stats from parsed graph
                dfg = parser.DFG
                nodes = len(dfg.dfg_nodes)
                edges = sum(len(e) for e in dfg.dfg_edges.values())
                
                if nodes > 0:
                    all_graphs[rel_path] = {
                        'nodes': nodes,
                        'edges': edges,
                        'node_names': [n.var_name for n in dfg.dfg_nodes.values()][:20]  # Sample
                    }
                    total_nodes += nodes
                    total_edges += edges
                    files_parsed += 1
                    
            except Exception as e:
                errors += 1
                if errors <= 5:
                    print(f"  Error in {rel_path}: {e}")
    
    # Save summary
    stats = {
        'total_nodes': total_nodes,
        'total_edges': total_edges,
        'files_parsed': files_parsed,
        'errors': errors,
        'sample_files': dict(list(all_graphs.items())[:10])
    }
    
    with open(output_path, 'wb') as f:
        pickle.dump(stats, f)
    
    print(f"  Nodes: {total_nodes}, Edges: {total_edges}, Files: {files_parsed}, Errors: {errors}")
    return stats


def main():
    repos_path = Path("/home/arsemkin/course-project-rag/swebench/repos")
    output_dir = Path("/home/arsemkin/course-project-rag/swebench/dataflow_graphs")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    repo_names = [
        'django',
        'sympy', 
        'scikit-learn',
        'matplotlib',
        'pytest',
        'astropy',
        'sphinx',
        'flask',
        'requests',
        'xarray',
        'seaborn',
        'pylint',
    ]
    
    all_stats = {}
    
    for repo_name in repo_names:
        repo_path = repos_path / repo_name
        if not repo_path.exists():
            print(f"Skipping {repo_name}: not found")
            continue
            
        output_path = output_dir / f"{repo_name}.pkl"
        stats = build_graph_for_repo(str(repo_path), str(output_path))
        all_stats[repo_name] = stats
    
    # Save summary
    with open(output_dir / "summary.json", 'w') as f:
        json.dump(all_stats, f, indent=2)
    
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    for repo, stats in all_stats.items():
        if 'error' in stats:
            print(f"{repo}: ERROR - {stats['error']}")
        else:
            print(f"{repo}: {stats['nodes']} nodes, {stats['edges']} edges")


if __name__ == "__main__":
    main()
