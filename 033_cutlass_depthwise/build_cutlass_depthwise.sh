#!/usr/bin/env bash
set -euo pipefail
CUTLASS_PATH="${CUTLASS_PATH:-$PWD/cutlass}"

if [[ ! -d "$CUTLASS_PATH" ]]; then
  git clone https://github.com/NVIDIA/cutlass.git "$CUTLASS_PATH"
fi

nvcc -O3 -arch=sm_89 -I "$CUTLASS_PATH/include" -I "$CUTLASS_PATH/tools/util/include" \
  cutlass_depthwise_conv.cu -o cutlass_depthwise_conv

./cutlass_depthwise_conv
