#!/usr/bin/env bash
set -euo pipefail

ARCH=${ARCH:-sm_90}

nvcc -O3 -lineinfo -arch="$ARCH" tma_fourier.cu -o tma_fourier
./tma_fourier 1024
