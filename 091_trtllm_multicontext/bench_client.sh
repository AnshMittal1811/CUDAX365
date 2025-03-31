#!/usr/bin/env bash
set -euo pipefail
TRTLLM_PATH="${TRTLLM_PATH:-$HOME/TensorRT-LLM}"
python "$TRTLLM_PATH/benchmarks/run_client.py" --host localhost --port 8000 --requests 100 --input_len 128 --output_len 128
