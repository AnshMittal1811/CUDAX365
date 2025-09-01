#!/usr/bin/env bash
set -euo pipefail

if ! command -v ncu >/dev/null 2>&1; then
  echo "ncu not found"
  exit 1
fi

if [[ ! -f ../243_reorder_latency_kernel/reorder_latency ]]; then
  (cd ../243_reorder_latency_kernel && ./run_reorder_latency.sh 1048576)
fi

ncu --metrics sm__inst_executed.avg,sm__cycles_elapsed.avg --target-processes all \
  ../243_reorder_latency_kernel/reorder_latency 1048576 | tee ipc_report.txt
