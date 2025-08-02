#!/usr/bin/env bash
set -euo pipefail

if [[ ! -f ../211_depth_aware_nerf/images.npy ]]; then
  (cd ../211_depth_aware_nerf && ./run_depth_nerf.sh)
fi

python compare_psnr.py --with-depth ../211_depth_aware_nerf/depths.npy --without-depth ../211_depth_aware_nerf/images.npy --out psnr_report.txt
