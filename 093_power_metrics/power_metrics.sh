#!/usr/bin/env bash
set -euo pipefail
python power_workload.py &
PID=$!
while kill -0 $PID 2>/dev/null; do
  nvidia-smi -q -d POWER | grep -E "Power Draw|Power Limit" >> power_log.txt
  sleep 1
done
wait $PID
