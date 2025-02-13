#!/usr/bin/env bash
set -euo pipefail
nvcc -O3 -arch=sm_89 -lineinfo lora_shuffle_reduce.cu -o lora_shuffle_reduce
./lora_shuffle_reduce
