#!/usr/bin/env bash
set -euo pipefail

ARCH=${ARCH:-sm_89}

nvcc -O3 -std=c++17 -arch=${ARCH} blackhole_flow_cuda.cu -o blackhole_flow_cuda
