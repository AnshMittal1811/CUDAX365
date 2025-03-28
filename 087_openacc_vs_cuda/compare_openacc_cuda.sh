#!/usr/bin/env bash
set -euo pipefail
nvc++ -acc -Minfo=accel ../086_openacc_flux/openacc_flux.c -o openacc_flux
nvcc -O3 -arch=sm_89 flux_cuda.cu -o flux_cuda

time ./openacc_flux

time ./flux_cuda
