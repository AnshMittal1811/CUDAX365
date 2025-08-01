#!/usr/bin/env bash
set -euo pipefail

ARCH=${ARCH:-sm_89}

nvcc -O3 -lineinfo -arch="$ARCH" depth_gather.cu -o depth_gather
./depth_gather 1024
