#!/usr/bin/env bash
set -euo pipefail
nvcc -O3 -arch=sm_89 -lineinfo cutensor_wmma.cu -lcutensor -o cutensor_wmma
./cutensor_wmma
