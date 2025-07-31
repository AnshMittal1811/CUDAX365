#!/usr/bin/env bash
set -euo pipefail

if [[ ! -f ../211_depth_aware_nerf/images.npy ]]; then
  (cd ../211_depth_aware_nerf && ./run_depth_nerf.sh)
fi

python generate_pseudo_lidar.py
python fuse_image_depth.py
