#!/usr/bin/env bash
set -euo pipefail
nvcc -O3 -arch=sm_89 -lineinfo scatter_shuffle.cu -o scatter_shuffle
./scatter_shuffle
