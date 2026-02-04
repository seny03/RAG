#!/bin/bash

# ===========================================
# Run CodeRAG Pipeline
# ===========================================

set -e

WORK_DIR="/home/arsemkin/course-project-rag/CodeRAG"
cd $WORK_DIR

# Активация виртуального окружения
source .venv/bin/activate

echo "=========================================="
echo "Running CodeRAG Pipeline"
echo "=========================================="

# Проверяем аргументы
STEP=${1:-"all"}

run_step() {
    echo ""
    echo "=========================================="
    echo "Running: $1"
    echo "=========================================="
    python scripts/$1
}

case $STEP in
    "query"|"1")
        run_step "build_query.py"
        ;;
    "retrieve"|"2")
        run_step "retrieve.py"
        ;;
    "rerank"|"3")
        run_step "rerank.py"
        ;;
    "prompt"|"4")
        run_step "build_prompt.py"
        ;;
    "inference"|"5")
        run_step "inference.py"
        ;;
    "eval"|"evaluation"|"6")
        run_step "evaluation.py"
        ;;
    "all")
        echo "Running full pipeline..."
        run_step "build_query.py"
        run_step "retrieve.py"
        run_step "rerank.py"
        run_step "build_prompt.py"
        run_step "inference.py"
        run_step "evaluation.py"
        ;;
    "no-rerank")
        echo "Running pipeline without rerank (faster)..."
        run_step "build_query.py"
        run_step "retrieve.py"
        # Skip rerank
        run_step "build_prompt.py"
        run_step "inference.py"
        run_step "evaluation.py"
        ;;
    *)
        echo "Usage: $0 [step]"
        echo ""
        echo "Steps:"
        echo "  query|1      - Build queries from code"
        echo "  retrieve|2   - Retrieve relevant code blocks"
        echo "  rerank|3     - Rerank retrieved code"
        echo "  prompt|4     - Build prompts for LLM"
        echo "  inference|5  - Run LLM inference"
        echo "  eval|6       - Evaluate results"
        echo "  all          - Run full pipeline (default)"
        echo "  no-rerank    - Run pipeline without reranking"
        exit 1
        ;;
esac

echo ""
echo "=========================================="
echo "Step '$STEP' completed!"
echo "=========================================="
