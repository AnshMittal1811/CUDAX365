#!/usr/bin/env bash
set -euo pipefail
nvcc -O3 -arch=sm_89 -ptx ptx_ref.cu -o ptx_ref.ptx
head -n 40 ptx_ref.ptx

