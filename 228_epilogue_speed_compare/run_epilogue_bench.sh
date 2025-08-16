#!/usr/bin/env bash
set -euo pipefail

ARCH=${ARCH:-sm_89}
W=${1:-128}
H=${2:-128}

nvcc -O3 -lineinfo -arch="$ARCH" epilogue_bench.cu -o epilogue_bench
./epilogue_bench "$W" "$H" | tee epilogue_bench.txt
