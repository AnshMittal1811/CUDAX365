#!/usr/bin/env bash
set -euo pipefail

ARCH=${ARCH:-sm_89}
N=${1:-1024}

nvcc -O3 -lineinfo -arch="$ARCH" reward_shaping.cu -o reward_shaping
./reward_shaping "$N"
