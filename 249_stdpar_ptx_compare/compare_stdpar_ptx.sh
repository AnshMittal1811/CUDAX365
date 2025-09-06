#!/usr/bin/env bash
set -euo pipefail

ARCH=${ARCH:-sm_89}

if command -v nvcc >/dev/null 2>&1; then
  nvcc -O3 -arch="$ARCH" -ptx hand_kernel.cu -o hand_kernel.ptx
fi

if command -v nvc++ >/dev/null 2>&1; then
  nvc++ -std=c++17 -stdpar -gpu=cc80,ptxinfo ../248_stdpar_offload/stdpar_example.cpp -c -o stdpar_example.o || true
fi

if [[ -f hand_kernel.ptx ]]; then
  echo "Hand PTX generated: hand_kernel.ptx"
fi
