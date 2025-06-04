#!/usr/bin/env bash
set -euo pipefail

python tensorboard_stability.py --logdir runs

echo "To view: tensorboard --logdir runs"
