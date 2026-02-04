#!/bin/bash

# ===========================================
# Download Benchmarks Script
# ===========================================

set -e

WORK_DIR="/home/arsemkin/course-project-rag"
cd $WORK_DIR

echo "=========================================="
echo "Downloading Benchmarks for CodeRAG"
echo "=========================================="

# 1. Скачивание CCEval
echo "[1/3] Downloading CCEval..."
if [ ! -d "cceval" ]; then
    git clone https://github.com/amazon-science/cceval.git
fi

# 2. Скачивание данных CCEval с Hugging Face
echo "[2/3] Downloading CCEval data from Hugging Face..."
mkdir -p CodeRAG/cache/benchmark/cceval

# CCEval использует данные с Hugging Face
python3 << 'EOF'
from huggingface_hub import snapshot_download
import os

# Скачиваем данные CCEval
try:
    snapshot_download(
        repo_id="Fsoft-AIC/CrossCodeEval",
        repo_type="dataset",
        local_dir="CodeRAG/cache/benchmark/cceval/hf_data",
        allow_patterns=["*.jsonl", "*.json", "**/*.py"]
    )
    print("CCEval data downloaded successfully!")
except Exception as e:
    print(f"Error downloading CCEval: {e}")
    print("You may need to download manually from https://huggingface.co/datasets/Fsoft-AIC/CrossCodeEval")
EOF

# 3. Скачивание ReccEval (если нужен)
echo "[3/3] Downloading ReccEval..."
# ReccEval обычно скачивается отдельно
# Проверьте: https://github.com/FSoft-AI4Code/ReccEval

python3 << 'EOF'
from huggingface_hub import snapshot_download
import os

try:
    # Попробуем скачать ReccEval
    snapshot_download(
        repo_id="Fsoft-AIC/ReccEval",
        repo_type="dataset",
        local_dir="CodeRAG/cache/benchmark/ReccEval",
        allow_patterns=["*.jsonl", "*.json", "**/*.py"]
    )
    print("ReccEval data downloaded successfully!")
except Exception as e:
    print(f"Note: ReccEval may not be available on HF: {e}")
    print("Check: https://github.com/FSoft-AI4Code/ReccEval")
EOF

echo "=========================================="
echo "Download complete!"
echo ""
echo "Next steps:"
echo "1. Check downloaded data in CodeRAG/cache/benchmark/"
echo "2. Update config paths if needed"
echo "3. Run the CodeRAG pipeline"
echo "=========================================="
