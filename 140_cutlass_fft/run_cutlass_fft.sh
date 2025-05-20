#!/usr/bin/env bash
set -euo pipefail

N=${1:-4096}
BATCH=${2:-64}
ARCH=${ARCH:-sm_89}

if [[ -n "${CUTLASS_DIR:-}" ]]; then
  if [[ -f "$CUTLASS_DIR/examples/fft/fft_1d.cu" ]]; then
    echo "Building CUTLASS FFT example from $CUTLASS_DIR/examples/fft/fft_1d.cu"
    nvcc -O3 -std=c++17 -arch="$ARCH" \
      -I"$CUTLASS_DIR/include" -I"$CUTLASS_DIR/tools/util/include" \
      "$CUTLASS_DIR/examples/fft/fft_1d.cu" -o cutlass_fft_1d
    ./cutlass_fft_1d --n "$N" --batch "$BATCH" || true
  else
    echo "CUTLASS FFT example not found. Set CUTLASS_DIR to a CUTLASS repo with fft examples."
  fi
else
  echo "CUTLASS_DIR not set. Running cuFFT benchmark only."
fi

nvcc -O3 -std=c++17 -arch="$ARCH" cufft_1d_bench.cu -lcufft -o cufft_1d_bench
./cufft_1d_bench "$N" "$BATCH"
