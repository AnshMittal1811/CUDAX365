#!/usr/bin/env bash
set -euo pipefail

ARCH=${ARCH:-sm_89}
BYTES=${1:-16777216}

nvcc -O3 -lineinfo -arch="$ARCH" p2p_compare.cu -o p2p_compare
./p2p_compare "$BYTES" | tee p2p_compare.txt
