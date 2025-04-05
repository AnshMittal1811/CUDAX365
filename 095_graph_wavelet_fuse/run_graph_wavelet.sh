#!/usr/bin/env bash
set -euo pipefail
nvcc -O3 -arch=sm_89 -lineinfo graph_wavelet.cu -o graph_wavelet
./graph_wavelet
