#!/usr/bin/env bash
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/scripts/cuda_mobile_env.sh"
ARCH=${ARCH:-sm_89}

"$NVCC" -O3 -lineinfo -arch="$ARCH" cp_async_bulk.cu -o cp_async_bulk
./cp_async_bulk 1024
