#!/usr/bin/env bash
set -euo pipefail

ARCH=${ARCH:-sm_89}
ITERS=${1:-1000}

nvcc -O3 -lineinfo -arch="$ARCH" event_blocking_overhead.cu -o event_blocking_overhead
./event_blocking_overhead "$ITERS" | tee event_overhead.txt
