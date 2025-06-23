#!/usr/bin/env bash
set -euo pipefail

if ! command -v nsys >/dev/null 2>&1; then
  echo "nsys not found"
  exit 1
fi

(cd ../174_mps_graph_partition && ./run_mps_graph.sh 1048576 100 &)
PID=$!
nsys profile -o mps_overlap_report --duration 5 --force-overwrite true --sample=none --trace=cuda,nvtx
wait "$PID" || true
