#!/usr/bin/env bash
set -euo pipefail
nvcc -O3 -arch=sm_89 pbo_isosurface.cu -lglfw -lGLEW -lGL -o pbo_isosurface
./pbo_isosurface
