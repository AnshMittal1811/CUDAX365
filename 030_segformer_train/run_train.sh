#!/usr/bin/env bash
set -euo pipefail
python train_segformer_b0.py --epochs 1 --batch-size 2
