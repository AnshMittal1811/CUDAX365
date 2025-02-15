#!/usr/bin/env bash
set -euo pipefail
TRTLLM_PATH="${TRTLLM_PATH:-$HOME/TensorRT-LLM}"
python "$TRTLLM_PATH/examples/llama/benchmark/benchmark.py" --engine_dir ./trtllm_engine --batch_size 1 --input_len 128 --output_len 128
