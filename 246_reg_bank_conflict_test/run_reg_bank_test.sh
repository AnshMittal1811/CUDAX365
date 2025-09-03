#!/usr/bin/env bash
set -euo pipefail

ARCH=${ARCH:-sm_89}
N=${1:-1048576}

nvcc -O3 -lineinfo -arch="$ARCH" reg_bank_kernel.cu -o reg_bank_kernel
./reg_bank_kernel "$N"

if command -v ncu >/dev/null 2>&1; then
  ncu --metrics smsp__sass_average_branch_targets_threads_uniform.pct --target-processes all \
    ./reg_bank_kernel "$N" | tee reg_bank_report.txt
fi
