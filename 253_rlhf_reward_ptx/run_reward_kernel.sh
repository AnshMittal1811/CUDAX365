#!/usr/bin/env bash
set -euo pipefail

ARCH=${ARCH:-sm_89}
N=${1:-1024}

nvcc -O3 -lineinfo -arch="$ARCH" reward_kernel.cu -o reward_kernel
./reward_kernel "$N"
