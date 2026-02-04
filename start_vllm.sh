#!/bin/bash
# Start vLLM server for inference

cd /home/arsemkin/course-project-rag

echo "Starting vLLM server on port 8001..."
echo "Model: Qwen/Qwen2.5-Coder-7B-Instruct"
echo ""

# Use smaller model for faster startup
CodeRAG/.venv/bin/python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen2.5-Coder-7B-Instruct \
    --port 8001 \
    --max-model-len 4096 \
    --gpu-memory-utilization 0.9

# Alternative: use even smaller model for testing
# CodeRAG/.venv/bin/python -m vllm.entrypoints.openai.api_server \
#     --model Qwen/Qwen2.5-Coder-1.5B-Instruct \
#     --port 8001
