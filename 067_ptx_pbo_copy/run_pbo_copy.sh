#!/usr/bin/env bash
set -euo pipefail
nvcc -O3 -arch=sm_89 -lineinfo pbo_copy_ptx.cu -o pbo_copy_ptx
./pbo_copy_ptx
