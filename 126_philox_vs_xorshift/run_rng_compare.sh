#!/usr/bin/env bash
set -euo pipefail
nvcc -O3 -arch=sm_89 -lineinfo philox_vs_xorshift.cu -lcurand -o rng_compare
./rng_compare
