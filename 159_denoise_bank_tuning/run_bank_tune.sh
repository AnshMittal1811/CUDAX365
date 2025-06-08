#!/usr/bin/env bash
set -euo pipefail

ARCH=${ARCH:-sm_89}
W=${1:-512}
H=${2:-512}

nvcc -O3 -lineinfo -arch="$ARCH" denoise_bank_tune.cu -o denoise_bank_tune
./denoise_bank_tune "$W" "$H"
