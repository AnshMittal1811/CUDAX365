#!/usr/bin/env bash
set -euo pipefail

ARCH=${ARCH:-sm_89}

if command -v nvcc >/dev/null 2>&1; then
  nvcc -O3 -arch="$ARCH" -ptx kernel_compare.cu -o nvcc_kernel.ptx
else
  echo "nvcc not found"
fi

if command -v nvc++ >/dev/null 2>&1; then
  nvc++ -acc -gpu=cc80,ptxinfo kernel_compare.cu -c -o openacc_kernel.o || true
else
  echo "nvc++ not found"
fi

if [[ -f nvcc_kernel.ptx ]]; then
  echo "NVCC PTX saved to nvcc_kernel.ptx"
fi
