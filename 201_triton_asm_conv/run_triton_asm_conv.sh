#!/usr/bin/env bash
set -euo pipefail

python triton_asm_conv.py
./find_triton_ptx.sh
