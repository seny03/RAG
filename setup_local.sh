#!/bin/bash

# ===========================================
# Local Setup Script for CodeRAG
# ===========================================

set -e

WORK_DIR="/home/arsemkin/course-project-rag"

echo "=========================================="
echo "CodeRAG Local Setup"
echo "=========================================="

# 1. Install uv if not present
echo "[1/5] Checking uv package manager..."
if ! command -v uv &> /dev/null; then
    echo "Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
else
    echo "uv already installed ✓"
fi

# 2. Setup CodeRAG
echo "[2/5] Setting up CodeRAG environment..."
cd $WORK_DIR/CodeRAG
uv sync

# 3. Create cache directories
echo "[3/5] Creating cache directories..."
mkdir -p cache/benchmark/cceval/{raw_data,python}
mkdir -p cache/benchmark/ReccEval/Source_Code
mkdir -p cache/cceval/{query,retrieve,rerank,prompt,inference,evaluation,dataflow/graphs}
mkdir -p cache/recceval/{query,retrieve,rerank,prompt,inference,evaluation,dataflow/graphs}
mkdir -p $WORK_DIR/swebench

# 4. Make scripts executable
echo "[4/5] Making scripts executable..."
cd $WORK_DIR
chmod +x setup_local.sh
chmod +x run_pipeline.sh
chmod +x download_benchmarks.sh
chmod +x download_repos.py
chmod +x analyze_swebench.py

# 5. Copy config
echo "[5/5] Setting up configuration..."
if [ ! -f "$WORK_DIR/CodeRAG/config/config.toml" ]; then
    cp $WORK_DIR/CodeRAG/config/config_server.toml $WORK_DIR/CodeRAG/config/config.toml
    echo "Config copied from config_server.toml"
fi

echo ""
echo "=========================================="
echo "Setup Complete! ✓"
echo "=========================================="
echo ""
echo "Next steps:"
echo ""
echo "1. Activate environment:"
echo "   cd $WORK_DIR/CodeRAG"
echo "   source .venv/bin/activate"
echo ""
echo "2. Download benchmarks (optional):"
echo "   cd $WORK_DIR"
echo "   ./download_benchmarks.sh"
echo ""
echo "3. Run pipeline:"
echo "   ./run_pipeline.sh query    # Just build queries"
echo "   ./run_pipeline.sh all      # Full pipeline"
echo ""
echo "Models will be auto-downloaded from HuggingFace on first run."
echo "=========================================="
