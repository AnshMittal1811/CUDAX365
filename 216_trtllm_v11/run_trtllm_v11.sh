#!/usr/bin/env bash
set -euo pipefail

./check_trtllm.sh

# Placeholder benchmark command for TRT-LLM v1.1
if command -v trtllm-runner >/dev/null 2>&1; then
  trtllm-runner --model-path ./model --max-output-len 32 --batch-size 1 > trtllm_run.txt || true
else
  echo "trtllm-runner not found" > trtllm_run.txt
fi
