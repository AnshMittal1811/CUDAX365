#!/usr/bin/env bash
set -euo pipefail
nvcc -O3 -arch=sm_89 -lineinfo cc_label_ptx.cu -o cc_label
./cc_label
