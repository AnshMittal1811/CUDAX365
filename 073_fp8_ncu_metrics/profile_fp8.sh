#!/usr/bin/env bash
set -euo pipefail
BIN="../072_segformer_fp8_cublaslt/fp8_gemm"
if [[ ! -x "$BIN" ]]; then
  echo "build fp8_gemm first"
  exit 1
fi
ncu --metrics sm__inst_executed_pipe_tensor.sum,sm__pipe_tensor_active.avg.pct $BIN || true
