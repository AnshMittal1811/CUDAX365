#!/usr/bin/env bash
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/scripts/cuda_mobile_env.sh"
ARCH=${ARCH:-sm_89}

"$NVCC" -O3 -lineinfo -arch="$ARCH" tma_fourier.cu -o tma_fourier
./tma_fourier 1024
