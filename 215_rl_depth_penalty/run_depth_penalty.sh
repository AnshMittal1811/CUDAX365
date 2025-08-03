#!/usr/bin/env bash
set -euo pipefail

if [[ ! -f ../211_depth_aware_nerf/depths.npy ]]; then
  (cd ../211_depth_aware_nerf && ./run_depth_nerf.sh)
fi

python rl_depth_penalty.py --depths ../211_depth_aware_nerf/depths.npy --out depth_penalty_rewards.csv
