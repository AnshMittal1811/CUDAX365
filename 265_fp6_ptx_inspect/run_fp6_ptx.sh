#!/usr/bin/env bash
set -euo pipefail

ARCH=${ARCH:-sm_89}

nvcc -O3 -lineinfo -arch="$ARCH" -ptx fp6_emulation.cu -o fp6_emulation.ptx

if command -v rg >/dev/null 2>&1; then
  rg -n "\.b8|\.s8|\.u8" fp6_emulation.ptx > fp6_ptx_inspect.txt || true
fi
