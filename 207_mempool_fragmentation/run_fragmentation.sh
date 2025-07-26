#!/usr/bin/env bash
set -euo pipefail

ARCH=${ARCH:-sm_89}

nvcc -O3 -lineinfo -arch="$ARCH" fragmentation_test.cu -o fragmentation_test
./fragmentation_test | tee fragmentation_log.txt
