#!/usr/bin/env bash
set -euo pipefail

ARCH=${ARCH:-sm_89}

nvcc -O3 -std=c++17 -arch=${ARCH} -lineinfo -Xptxas -v -use_fast_math \
  multi_bh_cuda.cu -o multi_bh_cuda
