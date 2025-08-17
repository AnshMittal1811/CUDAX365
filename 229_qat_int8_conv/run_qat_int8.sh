#!/usr/bin/env bash
set -euo pipefail

ARCH=${ARCH:-sm_89}

python qat_int8_conv.py

nvcc -O3 -lineinfo -arch="$ARCH" int8_conv.cu -o int8_conv
./int8_conv 4096

if command -v cuobjdump >/dev/null 2>&1; then
  cuobjdump --dump-ptx int8_conv | rg -n "dp4a" > dp4a_ptx.txt || true
fi
