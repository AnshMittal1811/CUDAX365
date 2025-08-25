#!/usr/bin/env bash
set -euo pipefail

if ! command -v ncu >/dev/null 2>&1; then
  echo "ncu not found"
  exit 1
fi

(cd ../236_tma_fourier_lookup && ./run_tma_fourier.sh)

ncu --metrics lts__t_sectors_hit_rate.pct --target-processes all ../236_tma_fourier_lookup/tma_fourier 1024 | tee tma_l2_report.txt
