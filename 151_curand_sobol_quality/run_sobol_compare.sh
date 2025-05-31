#!/usr/bin/env bash
set -euo pipefail

ARCH=${ARCH:-sm_89}
N=${1:-1048576}

nvcc -O3 -lineinfo -arch="$ARCH" sobol_compare.cu -lcurand -o sobol_compare
./sobol_compare "$N"
