#!/usr/bin/env bash
set -euo pipefail
nvcc -O3 -arch=sm_89 -Xptxas -v cub_reduce.cu -o cub_reduce 2> cub_regs.txt
python ../133_triton_reduction_asm/triton_reduce_asm.py
cat cub_regs.txt | grep -E "Used|registers" || true
