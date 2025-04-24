#!/usr/bin/env bash
set -euo pipefail
nvcc -O3 -arch=sm_89 -lineinfo dp4a_vs_hmma.cu -o dp4a_vs_hmma
./dp4a_vs_hmma
