#!/usr/bin/env bash
set -euo pipefail
nvcc -O3 -arch=sm_89 -lineinfo cutensor_4d.cu -lcutensor -o cutensor_4d
./cutensor_4d
