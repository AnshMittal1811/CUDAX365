#!/usr/bin/env bash
set -euo pipefail

if ! command -v mpirun >/dev/null 2>&1; then
  echo "mpirun not found"
  exit 1
fi

mpicxx -O3 mpi_domain_split.cpp -o mpi_domain_split
mpirun -np 2 ./mpi_domain_split | tee mpi_domain_split_log.txt
