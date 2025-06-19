#!/usr/bin/env bash
set -euo pipefail

ARCH=${ARCH:-sm_89}
N=${1:-4194304}

nvcc -O3 -lineinfo -arch="$ARCH" bench_kernel.cu -o bench_kernel
./bench_kernel "$N" | tee bench_results.txt
