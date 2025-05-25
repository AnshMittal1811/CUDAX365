#!/usr/bin/env bash
set -euo pipefail

ARCH=${ARCH:-sm_89}
MATS=${1:-8192}
TILES=${2:-1024}

nvcc -O3 -lineinfo -arch="$ARCH" cross_vs_wmma.cu -o cross_vs_wmma
./cross_vs_wmma "$MATS" "$TILES"
