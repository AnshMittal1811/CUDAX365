#!/usr/bin/env bash
set -euo pipefail

HOURS=${1:-6}
END=$(( $(date +%s) + HOURS * 3600 ))

while [[ $(date +%s) -lt $END ]]; do
  python ../200_stress_test_8h/pipeline_step.py >> stress_6h_log.txt
  if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi --query-gpu=timestamp,memory.used,utilization.gpu --format=csv >> stress_6h_gpu_log.csv
  fi
  sleep 10
done
