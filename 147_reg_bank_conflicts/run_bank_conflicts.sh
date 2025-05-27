#!/usr/bin/env bash
set -euo pipefail

ARCH=${ARCH:-sm_89}
THREADS=${THREADS:-256}
BLOCKS=${BLOCKS:-256}
ITERS=${ITERS:-1000}

nvcc -O3 -lineinfo -arch="$ARCH" bank_conflicts.cu -o bank_conflicts
./bank_conflicts "$THREADS" "$BLOCKS" "$ITERS"
