#!/usr/bin/env bash
set -euo pipefail

ARCH=${ARCH:-sm_89}
MATS=${1:-4096}

nvcc -O3 -lineinfo -arch="$ARCH" cross_product.cu -o cross_product
./cross_product "$MATS"
