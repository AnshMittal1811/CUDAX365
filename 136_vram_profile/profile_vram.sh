#!/usr/bin/env bash
set -euo pipefail
python ../135_fno_checkpointing/fno_checkpoint.py &
PID=$!
while kill -0 $PID 2>/dev/null; do
  nvidia-smi --query-gpu=memory.used --format=csv,noheader >> vram_log.csv
  sleep 1
done
wait $PID
