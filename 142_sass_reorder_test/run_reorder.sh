#!/usr/bin/env bash
set -euo pipefail

ARCH=${ARCH:-sm_89}
N=${1:-4194304}
BLOCK=${2:-256}

nvcc -O3 -lineinfo -arch="$ARCH" reorder_test.cu -o reorder_test
./reorder_test "$N" "$BLOCK"

if command -v cuobjdump >/dev/null 2>&1; then
  cuobjdump --dump-sass reorder_test > sass_dump.txt
  echo "Wrote sass_dump.txt"
fi
