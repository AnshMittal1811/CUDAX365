#!/usr/bin/env bash
set -euo pipefail

ARCH=${ARCH:-sm_89}

nvcc -O3 -lineinfo -arch="$ARCH" qgan_conv.cu -o qgan_conv

if command -v cuobjdump >/dev/null 2>&1; then
  cuobjdump --dump-sass qgan_conv > qgan_conv.sass.txt
  echo "Wrote qgan_conv.sass.txt"
else
  echo "cuobjdump not found"
fi
