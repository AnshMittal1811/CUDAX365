#!/usr/bin/env bash
set -euo pipefail
TRTLLM_PATH="${TRTLLM_PATH:-$HOME/TensorRT-LLM}"
ENGINE_DIR="${ENGINE_DIR:-$PWD/trtllm_engine}"
python "$TRTLLM_PATH/benchmarks/run_server.py" --engine_dir "$ENGINE_DIR" --num_contexts 4
