#!/usr/bin/env bash
set -euo pipefail

ARCH=${ARCH:-sm_90}

nvcc -O3 -lineinfo -arch="$ARCH" cp_async_bulk.cu -o cp_async_bulk
./cp_async_bulk 1024
