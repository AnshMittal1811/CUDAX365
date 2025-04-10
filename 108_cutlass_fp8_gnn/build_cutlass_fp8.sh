#!/usr/bin/env bash
set -euo pipefail
CUTLASS_PATH="${CUTLASS_PATH:-$PWD/cutlass}"

if [[ ! -d "$CUTLASS_PATH" ]]; then
  git clone https://github.com/NVIDIA/cutlass.git "$CUTLASS_PATH"
fi

nvcc -O3 -arch=sm_89 -I "$CUTLASS_PATH/include" -I "$CUTLASS_PATH/tools/util/include" \
  cutlass_fp8_gnn.cu -o cutlass_fp8_gnn

./cutlass_fp8_gnn
