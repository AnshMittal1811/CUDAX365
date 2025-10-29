#!/usr/bin/env bash
set -euo pipefail

if [[ ! -f ../301_satellite_dataset/sat_masks.npy ]]; then
  (cd ../301_satellite_dataset && ./run_satellite_dataset.sh)
fi

python finetune_segmentation.py
