#!/usr/bin/env bash
set -euo pipefail

ARCH=${ARCH:-sm_89}
W=${1:-256}
H=${2:-256}

nvcc -O3 -lineinfo -arch="$ARCH" fp8_conv_dp8a.cu -o fp8_conv_dp8a
./fp8_conv_dp8a "$W" "$H"
