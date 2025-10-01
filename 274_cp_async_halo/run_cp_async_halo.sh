#!/usr/bin/env bash
set -euo pipefail

ARCH=${ARCH:-sm_89}
N=${1:-1024}

nvcc -O3 -lineinfo -arch="$ARCH" cp_async_halo.cu -o cp_async_halo
./cp_async_halo "$N"
