#!/usr/bin/env bash
set -euo pipefail

if command -v trtllm-runner >/dev/null 2>&1; then
  trtllm-runner --version | tee trtllm_version.txt
else
  echo "trtllm-runner not found" | tee trtllm_version.txt
fi

