#!/usr/bin/env bash
set -euo pipefail

if [[ ! -f ../187_segformer_fp4_trt/mock_images.npy ]]; then
  (cd ../187_segformer_fp4_trt && ./run_fp4_quantize.sh)
fi

python segformer_accuracy.py --images ../187_segformer_fp4_trt/mock_images.npy --labels ../187_segformer_fp4_trt/mock_labels.npy --out accuracy_report.txt
