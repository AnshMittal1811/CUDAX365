#!/usr/bin/env bash
set -euo pipefail

ARCH=${ARCH:-sm_90}

nvcc -O3 -lineinfo -arch="$ARCH" mbarrier_async.cu -o mbarrier_async
./mbarrier_async 1024
