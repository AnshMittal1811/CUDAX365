#!/usr/bin/env bash
set -euo pipefail
nvcc -O3 -arch=sm_89 -lineinfo anneal_ptx.cu -o anneal_ptx
./anneal_ptx
