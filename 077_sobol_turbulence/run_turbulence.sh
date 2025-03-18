#!/usr/bin/env bash
set -euo pipefail
nvcc -O3 -arch=sm_89 -lineinfo sobol_turbulence.cu -lcurand -o sobol_turbulence
./sobol_turbulence
