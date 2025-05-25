#!/usr/bin/env bash
set -euo pipefail

ARCH=${ARCH:-sm_89}
N=${N:-16777216}
MIN_BLOCK=${MIN_BLOCK:-64}
MAX_BLOCK=${MAX_BLOCK:-1024}
STEP=${STEP:-32}

nvcc -O3 -lineinfo -arch="$ARCH" occupancy_tune.cu -o occupancy_tune
./occupancy_tune "$N" "$MIN_BLOCK" "$MAX_BLOCK" "$STEP"
