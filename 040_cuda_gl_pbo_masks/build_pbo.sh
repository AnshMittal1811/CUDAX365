#!/usr/bin/env bash
set -euo pipefail
nvcc -O3 -arch=sm_89 cuda_gl_pbo.cu -lglfw -lGLEW -lGL -o cuda_gl_pbo
./cuda_gl_pbo
