#!/usr/bin/env bash
set -euo pipefail

MINUTES=${1:-30}
END=$(( $(date +%s) + MINUTES * 60 ))

while [[ $(date +%s) -lt $END ]]; do
  python ../200_stress_test_8h/pipeline_step.py >> stability_log.txt
  if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi --query-gpu=timestamp,memory.used,utilization.gpu --format=csv >> stability_gpu_log.csv
  fi
  sleep 10
done
