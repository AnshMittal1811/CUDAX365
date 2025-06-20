#!/usr/bin/env bash
set -euo pipefail

if ! command -v nvc >/dev/null 2>&1 && ! command -v nvc++ >/dev/null 2>&1; then
  echo "NVC compiler not found"
  exit 1
fi

COMPILER=$(command -v nvc || command -v nvc++)

$COMPILER -acc -Minfo=accel -gpu=cc80,ptxinfo openacc_cutensor.c -o openacc_cutensor
./openacc_cutensor 1024
