#!/usr/bin/env bash
set -euo pipefail

ARCH=${ARCH:-sm_89}
N=${1:-1048576}

nvcc -O3 -lineinfo -arch="$ARCH" reorder_latency.cu -o reorder_latency
./reorder_latency "$N" | tee reorder_latency.txt
