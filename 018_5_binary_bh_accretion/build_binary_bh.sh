#!/usr/bin/env bash
set -euo pipefail

ARCH=${ARCH:-sm_89}

nvcc -O3 -std=c++17 -arch=${ARCH} -lineinfo -Xptxas -v -use_fast_math -rdc=true \
  binary_bh_sim.cu -lcufft -lcublas -lcurand -lcudadevrt -o binary_bh_sim
