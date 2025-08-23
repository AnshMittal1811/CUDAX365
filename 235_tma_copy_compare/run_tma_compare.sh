#!/usr/bin/env bash
set -euo pipefail

ARCH=${ARCH:-sm_90}
N=${1:-1048576}

nvcc -O3 -lineinfo -arch="$ARCH" copy_compare.cu -o copy_compare
./copy_compare "$N" | tee tma_copy_compare.txt
