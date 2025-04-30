#!/usr/bin/env bash
set -euo pipefail
nvcc -O3 -arch=sm_89 -lineinfo -Xptxas -v launch_bounds_conv.cu -o launch_bounds_conv
./launch_bounds_conv
