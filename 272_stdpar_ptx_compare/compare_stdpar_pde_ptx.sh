#!/usr/bin/env bash
set -euo pipefail

ARCH=${ARCH:-sm_89}

if command -v nvc++ >/dev/null 2>&1; then
  nvc++ -std=c++17 -stdpar -gpu=cc80,ptxinfo ../271_stdpar_pde_solver/stdpar_pde.cpp -c -o stdpar_pde.o || true
fi

if command -v nvcc >/dev/null 2>&1; then
  nvcc -O3 -arch="$ARCH" -ptx cuda_pde_kernel.cu -o cuda_pde_kernel.ptx
fi

if [[ -f cuda_pde_kernel.ptx ]]; then
  echo "CUDA PTX saved to cuda_pde_kernel.ptx" > ptx_compare.txt
else
  echo "CUDA PTX missing" > ptx_compare.txt
fi
