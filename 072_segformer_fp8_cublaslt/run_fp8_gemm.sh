#!/usr/bin/env bash
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/scripts/cuda_mobile_env.sh"
"$NVCC" -O3 -arch="${ARCH:-sm_89}" -lineinfo fp8_gemm_cublaslt.cu -lcublasLt -lcublas -o fp8_gemm
./fp8_gemm
