#!/usr/bin/env bash
set -euo pipefail

ARCH=${ARCH:-sm_89}
N=${N:-16777216}
BLOCK=${BLOCK:-256}
STRIDE=${STRIDE:-4}

nvcc -O3 -lineinfo -arch="$ARCH" stalls_kernel.cu -o stalls_kernel
./stalls_kernel "$N" "$BLOCK" "$STRIDE"

if command -v ncu >/dev/null 2>&1; then
  echo "Collecting Nsight Compute SourceCounters..."
  ncu --section SourceCounters --section SpeedOfLight --target-processes all \
    ./stalls_kernel "$N" "$BLOCK" "$STRIDE" | tee ncu_source_report.txt
else
  echo "ncu not found; skipping profiling."
fi
