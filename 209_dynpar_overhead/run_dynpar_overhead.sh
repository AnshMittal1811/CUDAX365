#!/usr/bin/env bash
set -euo pipefail

ARCH=${ARCH:-sm_89}

nvcc -O3 -lineinfo -rdc=true -arch="$ARCH" dynpar_overhead.cu -lcudadevrt -o dynpar_overhead
./dynpar_overhead 256 256 | tee dynpar_overhead.txt
