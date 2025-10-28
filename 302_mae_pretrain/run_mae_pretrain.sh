#!/usr/bin/env bash
set -euo pipefail

if [[ ! -f ../301_satellite_dataset/sat_images.npy ]]; then
  (cd ../301_satellite_dataset && ./run_satellite_dataset.sh)
fi

python pretrain_mae.py
