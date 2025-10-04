#!/usr/bin/env bash
set -euo pipefail

ARCH=${ARCH:-sm_89}

nvcc -O3 -lineinfo -arch="$ARCH" -ptx mixed_precision_kernel.cu -o mixed_precision_kernel.ptx

if command -v rg >/dev/null 2>&1; then
  rg -n "f16|half" mixed_precision_kernel.ptx > mixed_precision_ptx.txt || true
fi
