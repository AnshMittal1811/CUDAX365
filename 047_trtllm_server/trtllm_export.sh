#!/usr/bin/env bash
set -euo pipefail
TRTLLM_PATH="${TRTLLM_PATH:-$HOME/TensorRT-LLM}"
MODEL_PATH="${MODEL_PATH:-$HOME/models/TinyLlama}"

python "$TRTLLM_PATH/examples/llama/convert_checkpoint.py" \
  --model_dir "$MODEL_PATH" --output_dir ./trtllm_ckpt

python "$TRTLLM_PATH/examples/llama/build.py" \
  --checkpoint_dir ./trtllm_ckpt --output_dir ./trtllm_engine
