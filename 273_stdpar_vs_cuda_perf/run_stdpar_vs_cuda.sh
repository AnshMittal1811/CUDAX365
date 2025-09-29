#!/usr/bin/env bash
set -euo pipefail

ARCH=${ARCH:-sm_89}

if command -v nvc++ >/dev/null 2>&1; then
  (cd ../271_stdpar_pde_solver && ./build_stdpar_pde.sh) | tee stdpar_run.txt
else
  echo "nvc++ not found" > stdpar_run.txt
fi

nvcc -O3 -lineinfo -arch="$ARCH" cuda_pde_solver.cu -o cuda_pde_solver
./cuda_pde_solver 1048576 | tee cuda_run.txt
