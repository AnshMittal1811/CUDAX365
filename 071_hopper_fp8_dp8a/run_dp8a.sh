#!/usr/bin/env bash
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/scripts/cuda_mobile_env.sh"
"$NVCC" -O3 -arch="${ARCH:-sm_89}" -lineinfo dp8a_dummy.cu -o dp8a_dummy
./dp8a_dummy
