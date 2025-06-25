#!/usr/bin/env bash
set -euo pipefail

ARCH=${ARCH:-sm_89}
ITERS=${1:-10000}

nvcc -O3 -lineinfo -arch="$ARCH" latency_bench.cu -o latency_bench
./latency_bench "$ITERS" | tee latency_results.txt
