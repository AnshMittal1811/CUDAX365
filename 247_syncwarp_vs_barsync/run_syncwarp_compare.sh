#!/usr/bin/env bash
set -euo pipefail

ARCH=${ARCH:-sm_89}
ITERS=${1:-1000}

nvcc -O3 -lineinfo -arch="$ARCH" syncwarp_compare.cu -o syncwarp_compare
./syncwarp_compare "$ITERS" | tee syncwarp_compare.txt
