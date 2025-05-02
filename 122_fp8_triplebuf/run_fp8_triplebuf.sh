#!/usr/bin/env bash
set -euo pipefail
nvcc -O3 -arch=sm_89 -lineinfo fp8_triplebuf.cu -o fp8_triplebuf
./fp8_triplebuf
