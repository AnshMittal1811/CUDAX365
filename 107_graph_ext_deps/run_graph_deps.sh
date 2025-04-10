#!/usr/bin/env bash
set -euo pipefail
nvcc -O3 -arch=sm_89 -lineinfo graph_ext_deps.cu -o graph_ext_deps
./graph_ext_deps
