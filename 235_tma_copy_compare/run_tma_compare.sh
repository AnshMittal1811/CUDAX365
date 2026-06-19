#!/usr/bin/env bash
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/scripts/cuda_mobile_env.sh"
ARCH=${ARCH:-sm_89}
N=${1:-1048576}

"$NVCC" -O3 -lineinfo -arch="$ARCH" copy_compare.cu -o copy_compare
./copy_compare "$N" | tee tma_copy_compare.txt
