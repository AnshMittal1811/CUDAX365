#!/usr/bin/env bash
set -euo pipefail

ARCH=${ARCH:-sm_89}
W=${1:-64}
H=${2:-64}

nvcc -O3 -lineinfo -arch="$ARCH" cute_conv.cu -o cute_conv
./cute_conv "$W" "$H"

nvcc -O3 -lineinfo -arch="$ARCH" -ptx cute_conv.cu -o cute_conv.ptx

if command -v cuobjdump >/dev/null 2>&1; then
  cuobjdump --dump-ptx cute_conv > cute_conv_dump.ptx || true
fi

if [[ -n "${CUTLASS_DIR:-}" ]]; then
  echo "CUTLASS_DIR set to $CUTLASS_DIR; hook cuTe conv here if available."
fi
