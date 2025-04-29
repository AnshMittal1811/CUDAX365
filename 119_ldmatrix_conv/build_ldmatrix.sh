#!/usr/bin/env bash
set -euo pipefail
nvcc -O3 -arch=sm_89 -lineinfo -ptx ldmatrix_conv.cu -o ldmatrix_conv.ptx
rg -n "ldmatrix" ldmatrix_conv.ptx || grep -n "ldmatrix" ldmatrix_conv.ptx || true
