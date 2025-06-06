#!/usr/bin/env bash
set -euo pipefail

ARCH=${ARCH:-sm_89}
N=${1:-1048576}

nvcc -O3 -lineinfo -arch="$ARCH" wavelet_ptx.cu -o wavelet_ptx
./wavelet_ptx "$N"
