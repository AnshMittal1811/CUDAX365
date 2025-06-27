#!/usr/bin/env bash
set -euo pipefail

if ! command -v ncu >/dev/null 2>&1; then
  echo "ncu not found"
  exit 1
fi

(cd ../178_async_memcpy_overlap && ./run_async_copy.sh 4194304)

ncu --metrics lts__t_sectors_hit_rate.pct --target-processes all ../178_async_memcpy_overlap/async_memcpy 4194304 | tee l2_hit_report.txt
