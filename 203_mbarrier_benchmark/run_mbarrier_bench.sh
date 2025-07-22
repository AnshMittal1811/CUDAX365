#!/usr/bin/env bash
set -euo pipefail

ARCH=${ARCH:-sm_90}
ITERS=${1:-1000}

nvcc -O3 -lineinfo -arch="$ARCH" mbarrier_bench.cu -o mbarrier_bench
./mbarrier_bench "$ITERS" | tee mbarrier_bench.txt
