#!/usr/bin/env bash
set -euo pipefail

ARCH=${ARCH:-sm_89}
BLOCK=${BLOCK:-256}
N=${N:-1048576}

nvcc -O3 -lineinfo -arch="$ARCH" sm90_occ_kernel.cu -o sm90_occ
./sm90_occ "$N" "$BLOCK"

if command -v ncu >/dev/null 2>&1; then
  echo "Profiling with Nsight Compute..."
  ncu --set full --csv --target-processes all ./sm90_occ "$N" "$BLOCK" | tee ncu_report.csv
else
  echo "ncu not found; skipping profiling."
fi
