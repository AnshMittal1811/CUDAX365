#!/usr/bin/env bash
set -euo pipefail

ARCH=${ARCH:-sm_89}
N=${1:-4096}

nvcc -O3 -lineinfo -arch="$ARCH" fp8_dropout.cu -o fp8_dropout
./fp8_dropout "$N"
