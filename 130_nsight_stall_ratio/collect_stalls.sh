#!/usr/bin/env bash
set -euo pipefail
BIN=${1:-../114_dp4a_vs_hmma/dp4a_vs_hmma}
if [[ ! -x "$BIN" ]]; then
  echo "build dp4a_vs_hmma first"
  exit 1
fi
ncu --csv --metrics smsp__sass_average_branch_targets_threads_ratio,smsp__sass_average_branch_targets_threads_per_warp_active.pct \
  "$BIN" > stalls.csv || true
