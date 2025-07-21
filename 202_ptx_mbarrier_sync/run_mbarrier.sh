#!/usr/bin/env bash
set -euo pipefail

ARCH=${ARCH:-sm_90}

nvcc -O3 -lineinfo -arch="$ARCH" mbarrier_test.cu -o mbarrier_test
./mbarrier_test
