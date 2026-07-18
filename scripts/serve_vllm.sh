#!/usr/bin/env bash
# Start a local vLLM OpenAI-compatible server for this project (L4-friendly defaults).
#
# Usage:
#   ./scripts/serve_vllm.sh Qwen/Qwen3-4B
#   ./scripts/serve_vllm.sh Qwen/Qwen3-4B 8000 4096
#
# Then in another terminal:
#   export OPENAI_COMPATIBLE_BASE_URL="http://localhost:8000/v1"
#   export OPENAI_COMPATIBLE_API_KEY="EMPTY"
#   uv run python scripts/run_inference.py --config configs/e1_reasoning_language_vllm.json ...

set -euo pipefail

MODEL="${1:-Qwen/Qwen3-4B}"
PORT="${2:-8000}"
MAX_MODEL_LEN="${3:-4096}"
HOST="${HOST:-0.0.0.0}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.90}"

echo "Serving ${MODEL} on ${HOST}:${PORT} (max-model-len=${MAX_MODEL_LEN})"
echo "FlashAttention: enabled automatically on L4/Ada (SM>=8) when available in vLLM."
echo "Client env:"
echo "  export OPENAI_COMPATIBLE_BASE_URL=\"http://localhost:${PORT}/v1\""
echo "  export OPENAI_COMPATIBLE_API_KEY=\"EMPTY\""

exec vllm serve "${MODEL}" \
  --host "${HOST}" \
  --port "${PORT}" \
  --dtype auto \
  --max-model-len "${MAX_MODEL_LEN}" \
  --gpu-memory-utilization "${GPU_MEM_UTIL}" \
  --enable-chunked-prefill \
  --disable-log-requests
