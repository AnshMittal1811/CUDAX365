#!/usr/bin/env bash
set -euo pipefail
nvcc -O3 -arch=sm_89 -lineinfo launch_bounds.cu -o launch_bounds
./launch_bounds
