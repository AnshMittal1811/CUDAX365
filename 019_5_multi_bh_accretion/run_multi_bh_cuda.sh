#!/usr/bin/env bash
set -euo pipefail

# 1-minute simulation @20 FPS => 1200 frames
./build_multi_bh_cuda.sh
./multi_bh_cuda --n-bh 8 --points-per-bh 384 --frames 1200 --dt 0.05 --out frames_multi
python animate_multi_bh_cuda.py --fps 20 --out multi_bh_accretion.mp4
