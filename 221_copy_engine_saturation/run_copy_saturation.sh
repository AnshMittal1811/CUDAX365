#!/usr/bin/env bash
set -euo pipefail

ARCH=${ARCH:-sm_89}
N=${1:-4194304}
STREAMS=${2:-4}

nvcc -O3 -lineinfo -arch="$ARCH" copy_saturation.cu -o copy_saturation
./copy_saturation "$N" "$STREAMS"
