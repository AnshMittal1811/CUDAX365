#!/usr/bin/env bash
set -euo pipefail

if ! command -v nvc++ >/dev/null 2>&1; then
  echo "nvc++ not found"
  exit 1
fi

nvc++ -std=c++17 -stdpar -gpu=cc80,ptxinfo stdpar_pde.cpp -o stdpar_pde
./stdpar_pde
