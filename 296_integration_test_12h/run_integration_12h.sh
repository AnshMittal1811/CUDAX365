#!/usr/bin/env bash
set -euo pipefail

HOURS=${1:-12}
END=$(( $(date +%s) + HOURS * 3600 ))

while [[ $(date +%s) -lt $END ]]; do
  python ../200_stress_test_8h/pipeline_step.py >> integration_12h_log.txt
  if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi --query-gpu=timestamp,memory.used,utilization.gpu --format=csv >> integration_12h_gpu_log.csv
  fi
  sleep 10
done
