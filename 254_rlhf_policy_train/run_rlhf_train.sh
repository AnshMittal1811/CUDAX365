#!/usr/bin/env bash
set -euo pipefail

if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi --query-gpu=timestamp,memory.used,utilization.gpu --format=csv -l 1 > vram_log.csv &
  LOGGER=$!
fi

python train_rlhf_policy.py

if [[ -n "${LOGGER:-}" ]]; then
  kill "$LOGGER" || true
fi
