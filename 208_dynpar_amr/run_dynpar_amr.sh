#!/usr/bin/env bash
set -euo pipefail

ARCH=${ARCH:-sm_89}

nvcc -O3 -lineinfo -rdc=true -arch="$ARCH" dynpar_amr.cu -lcudadevrt -o dynpar_amr
./dynpar_amr 256 256
