#!/bin/bash
# Run RepoBench evaluation
# Make sure vLLM server is running first!

cd /home/arsemkin/course-project-rag/repobench_eval

echo "=========================================="
echo "RepoBench Evaluation"
echo "=========================================="
echo ""
echo "Checking if vLLM server is running..."
curl -s http://localhost:8001/v1/models > /dev/null 2>&1
if [ $? -eq 0 ]; then
    echo "✓ vLLM server is running"
else
    echo "✗ vLLM server is NOT running!"
    echo ""
    echo "Start it with: ./start_vllm.sh"
    exit 1
fi

echo ""
echo "Running evaluation..."
../CodeRAG/.venv/bin/python run_repoeval.py

echo ""
echo "=========================================="
echo "Done! Check results in:"
echo "  repoeval_results.json"
echo "=========================================="
