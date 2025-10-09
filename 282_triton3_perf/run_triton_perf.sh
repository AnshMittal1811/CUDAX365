#!/usr/bin/env bash
set -euo pipefail

if [[ ! -f ../201_triton_asm_conv/triton_conv_log.txt ]]; then
  (cd ../201_triton_asm_conv && ./run_triton_asm_conv.sh)
fi

if [[ ! -f ../281_triton3_port/triton3_log.txt ]]; then
  (cd ../281_triton3_port && ./run_triton3.sh)
fi

python compare_triton_perf.py
