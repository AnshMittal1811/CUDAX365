#!/usr/bin/env bash
set -euo pipefail

ARCH=${ARCH:-sm_89}
W=${1:-128}
H=${2:-128}

nvcc -O3 -lineinfo -arch="$ARCH" inline_activation.cu -o inline_activation
./inline_activation "$W" "$H"
