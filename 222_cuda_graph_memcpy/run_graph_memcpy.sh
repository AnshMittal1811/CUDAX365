#!/usr/bin/env bash
set -euo pipefail

ARCH=${ARCH:-sm_89}
N=${1:-1048576}

nvcc -O3 -lineinfo -arch="$ARCH" graph_memcpy.cu -o graph_memcpy
./graph_memcpy "$N"
