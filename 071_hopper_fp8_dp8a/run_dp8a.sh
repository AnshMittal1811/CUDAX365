#!/usr/bin/env bash
set -euo pipefail
nvcc -O3 -arch=sm_90 -lineinfo dp8a_dummy.cu -o dp8a_dummy
./dp8a_dummy
