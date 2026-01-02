#!/usr/bin/env bash
set -euo pipefail

# 2-minute simulation @12 FPS => 1440 frames
# Reduce --frames or --fps for a faster preview.
./build_binary_bh.sh
./binary_bh_sim --nx 192 --ny 192 --frames 1440 --dt 0.05 --out frames_binary
python animate_binary_bh.py --shape 192 192 --stride 2 --fps 12 --out binary_bh_3d.mp4
