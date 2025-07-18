#!/usr/bin/env bash
set -euo pipefail

HOURS=${1:-8}
END=$(( $(date +%s) + HOURS * 3600 ))

while [[ $(date +%s) -lt $END ]]; do
  python pipeline_step.py >> stress_log.txt
  if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi --query-gpu=timestamp,utilization.gpu,memory.used --format=csv >> gpu_mem_log.csv
  fi
  sleep 10
done

