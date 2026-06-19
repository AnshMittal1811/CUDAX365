#!/usr/bin/env bash
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/scripts/cuda_mobile_env.sh"
ARCH=${ARCH:-sm_89}

"$NVCC" -O3 -lineinfo -arch="$ARCH" mbarrier_test.cu -o mbarrier_test
./mbarrier_test
