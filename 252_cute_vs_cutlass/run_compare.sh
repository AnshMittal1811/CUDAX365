#!/usr/bin/env bash
set -euo pipefail

ARCH=${ARCH:-sm_89}

if [[ ! -f ../251_cute_conv/cute_conv.ptx ]]; then
  (cd ../251_cute_conv && ./run_cute_conv.sh)
fi

nvcc -O3 -lineinfo -arch="$ARCH" -ptx baseline_conv.cu -o baseline_conv.ptx
python compare_cute_cutlass.py
