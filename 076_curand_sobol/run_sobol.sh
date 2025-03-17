#!/usr/bin/env bash
set -euo pipefail
nvcc -O3 -arch=sm_89 -lineinfo curand_sobol.cu -lcurand -o curand_sobol
./curand_sobol
