#!/usr/bin/env bash
set -euo pipefail

ARCH=${ARCH:-sm_89}
N=${1:-1048576}

nvcc -O3 -lineinfo -arch="$ARCH" xorshift_ptx.cu -o xorshift_ptx
./xorshift_ptx "$N"
