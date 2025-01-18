#!/usr/bin/env bash
set -euo pipefail

NX="${1:-128}"
NY="${2:-128}"
STEPS="${3:-50}"
REFINE="${4:-0.15}"

SRC="../017_dynpar_mhd/017_dynpar_mhd.cu"

nvcc -O3 -arch=sm_89 -rdc=true -lineinfo -Xptxas -v -DBLOCK_X=8  -DBLOCK_Y=8  "$SRC" -lcudadevrt -lcufft -o mhd_dynpar_8
nvcc -O3 -arch=sm_89 -rdc=true -lineinfo -Xptxas -v -DBLOCK_X=16 -DBLOCK_Y=16 "$SRC" -lcudadevrt -lcufft -o mhd_dynpar_16
nvcc -O3 -arch=sm_89 -rdc=true -lineinfo -Xptxas -v -DBLOCK_X=32 -DBLOCK_Y=8  "$SRC" -lcudadevrt -lcufft -o mhd_dynpar_32x8

echo "Running NCU (block 8x8)..."
ncu --set full --kernel-name regex:step_mhd_rusanov_dynpar --target-processes all \
  ./mhd_dynpar_8 "$NX" "$NY" "$STEPS" "$REFINE" 0

echo "Running NCU (block 16x16)..."
ncu --set full --kernel-name regex:step_mhd_rusanov_dynpar --target-processes all \
  ./mhd_dynpar_16 "$NX" "$NY" "$STEPS" "$REFINE" 0

echo "Running NCU (block 32x8)..."
ncu --set full --kernel-name regex:step_mhd_rusanov_dynpar --target-processes all \
  ./mhd_dynpar_32x8 "$NX" "$NY" "$STEPS" "$REFINE" 0
