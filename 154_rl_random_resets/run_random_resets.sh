#!/usr/bin/env bash
set -euo pipefail

python rl_random_resets.py --episodes 200 --max-steps 200 --reset-prob 0.15 --out rewards_random_resets.csv
