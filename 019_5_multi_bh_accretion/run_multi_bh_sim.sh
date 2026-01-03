#!/usr/bin/env bash
set -euo pipefail

# 1-minute simulation @20 FPS => 1200 frames
# Reduce --frames or --fps for a faster preview.
if command -v nvcc >/dev/null 2>&1; then
  ./run_multi_bh_cuda.sh
else
  python simulate_multi_bh.py --n-bh 8 --frames 1200 --fps 20 --dt 0.05 --points-per-bh 128 --out multi_bh_accretion.mp4
fi
