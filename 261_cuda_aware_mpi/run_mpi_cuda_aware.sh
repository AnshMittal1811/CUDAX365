#!/usr/bin/env bash
set -euo pipefail

if ! command -v mpirun >/dev/null 2>&1; then
  echo "mpirun not found"
  exit 1
fi

if ! command -v nvcc >/dev/null 2>&1; then
  echo "nvcc not found"
  exit 1
fi

nvcc -O3 -ccbin mpicxx mpi_cuda_aware.cpp -o mpi_cuda_aware -lmpi
mpirun -np 2 ./mpi_cuda_aware | tee mpi_cuda_aware_log.txt
