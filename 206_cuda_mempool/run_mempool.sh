#!/usr/bin/env bash
set -euo pipefail

ARCH=${ARCH:-sm_89}

nvcc -O3 -lineinfo -arch="$ARCH" mempool_demo.cu -o mempool_demo
./mempool_demo
