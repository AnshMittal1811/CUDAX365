#!/usr/bin/env bash
set -euo pipefail
nvcc -O3 -arch=sm_89 -lineinfo wavelet_ptx.cu -o wavelet_ptx
./wavelet_ptx
