#!/usr/bin/env bash
set -euo pipefail

if [[ ! -f ../187_segformer_fp4_trt/mock_labels.npy ]]; then
  (cd ../187_segformer_fp4_trt && ./run_fp4_quantize.sh)
fi

python lora_fp4_recover.py --labels ../187_segformer_fp4_trt/mock_labels.npy --out lora_recover.txt
