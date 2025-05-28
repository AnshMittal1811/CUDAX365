#!/usr/bin/env bash
set -euo pipefail

ARCH=${ARCH:-sm_89}
NX=${1:-512}
NY=${2:-512}
DT=${3:-0.1}

nvcc -O3 -lineinfo -arch="$ARCH" flux_kernel.cu -o flux_kernel
./flux_kernel "$NX" "$NY" "$DT" | tee flux_bench.txt
