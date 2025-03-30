#!/usr/bin/env bash
set -euo pipefail
nvcc -O3 -arch=sm_89 -lineinfo bar_sync_ptx.cu -o bar_sync_ptx
./bar_sync_ptx
