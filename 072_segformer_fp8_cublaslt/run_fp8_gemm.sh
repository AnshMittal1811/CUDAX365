#!/usr/bin/env bash
set -euo pipefail
nvcc -O3 -arch=sm_90 -lineinfo fp8_gemm_cublaslt.cu -lcublasLt -lcublas -o fp8_gemm
./fp8_gemm
