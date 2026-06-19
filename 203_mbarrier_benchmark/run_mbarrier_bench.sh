#!/usr/bin/env bash
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/scripts/cuda_mobile_env.sh"
ARCH=${ARCH:-sm_89}
ITERS=${1:-1000}

"$NVCC" -O3 -lineinfo -arch="$ARCH" mbarrier_bench.cu -o mbarrier_bench
./mbarrier_bench "$ITERS" | tee mbarrier_bench.txt
