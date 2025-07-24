#!/usr/bin/env bash
set -euo pipefail

if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "nvidia-smi not found"
  exit 1
fi

( cd ../204_int4_gnn_pde_nerf && ./run_pipeline.sh ) &
PID=$!

nvidia-smi --query-gpu=timestamp,memory.used,utilization.gpu --format=csv -l 1 > vram_log.csv &
LOGGER=$!

wait "$PID" || true
kill "$LOGGER" || true

