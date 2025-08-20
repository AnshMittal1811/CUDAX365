#!/usr/bin/env bash
set -euo pipefail

ARCH=${ARCH:-sm_89}

nvcc -O3 -lineinfo -arch="$ARCH" -ptx recompile_kernel.cu -o recompile_kernel.ptx

PTX_VERSION=$(rg -n "^\.version" recompile_kernel.ptx || true)

echo "$PTX_VERSION" > ptx_version.txt

if command -v cuobjdump >/dev/null 2>&1; then
  nvcc -O3 -lineinfo -arch="$ARCH" recompile_kernel.cu -o recompile_kernel
  cuobjdump --dump-sass recompile_kernel > recompile_kernel.sass.txt || true
fi
