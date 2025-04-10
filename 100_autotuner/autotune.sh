#!/usr/bin/env bash
set -euo pipefail
for block in 128 256 512; do
  for rreg in 32 64 128; do
    echo "block=$block rreg=$rreg"
    nvcc -O3 -arch=sm_89 -maxrregcount=$rreg autotune_kernel.cu -o autotune_kernel
    /usr/bin/time -f "%e" ./autotune_kernel $block
  done
done
