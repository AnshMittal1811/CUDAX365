#!/usr/bin/env bash
set -euo pipefail

python generate_depth_data.py
python train_depth_nerf.py --images images.npy --depths depths.npy --out depth_nerf_log.txt
