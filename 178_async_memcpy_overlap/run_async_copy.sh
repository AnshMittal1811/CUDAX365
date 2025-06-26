#!/usr/bin/env bash
set -euo pipefail

ARCH=${ARCH:-sm_89}
N=${1:-4194304}

nvcc -O3 -lineinfo -arch="$ARCH" async_memcpy.cu -o async_memcpy
./async_memcpy "$N" | tee async_copy_results.txt
