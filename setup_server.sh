#!/bin/bash

# ===========================================
# CodeRAG Setup Script
# ===========================================

set -e

echo "=========================================="
echo "CodeRAG Setup"
echo "=========================================="

# Настройка директории
WORK_DIR="/home/arsemkin/course-project-rag"
cd $WORK_DIR

# 1. Установка uv (package manager)
echo "[1/6] Installing uv package manager..."
if ! command -v uv &> /dev/null; then
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
    echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.bashrc
fi

# 2. Переход в директорию CodeRAG
echo "[2/6] Setting up CodeRAG..."
cd $WORK_DIR/CodeRAG

# 3. Синхронизация зависимостей
echo "[3/6] Installing dependencies with uv..."
uv sync

# 4. Создание директорий для кэша
echo "[4/6] Creating cache directories..."
mkdir -p cache/benchmark/cceval/raw_data
mkdir -p cache/benchmark/cceval/python
mkdir -p cache/benchmark/ReccEval/Source_Code
mkdir -p cache/cceval/{query,retrieve,rerank,prompt,inference,evaluation}
mkdir -p cache/recceval/{query,retrieve,rerank,prompt,inference,evaluation,dataflow/graphs}

# 5. Скачивание бенчмарка CCEval
echo "[5/6] Downloading CCEval benchmark..."
cd $WORK_DIR
if [ ! -d "cceval" ]; then
    git clone https://github.com/amazon-science/cceval.git
fi

# Копирование данных CCEval в cache
if [ -d "cceval/data" ]; then
    echo "Copying CCEval data..."
    # Данные нужно будет скачать отдельно или запросить у авторов
fi

# 6. Скачивание моделей (опционально - можно использовать Hugging Face)
echo "[6/6] Models setup..."
echo "Models will be downloaded from Hugging Face on first run."
echo ""
echo "Required models:"
echo "  - codet5p-220m (for query building)"
echo "  - codet5p-110m-embedding (for dense retrieval)"
echo "  - Qwen2.5-Coder-7B or similar (for inference)"
echo ""

# Активация виртуального окружения
echo "=========================================="
echo "Setup complete!"
echo ""
echo "To activate environment:"
echo "  cd $WORK_DIR/CodeRAG"
echo "  source .venv/bin/activate"
echo ""
echo "To run the pipeline:"
echo "  python scripts/build_query.py"
echo "  python scripts/retrieve.py"
echo "  python scripts/rerank.py"
echo "  python scripts/build_prompt.py"
echo "  python scripts/inference.py"
echo "  python scripts/evaluation.py"
echo "=========================================="
