#!/usr/bin/env bash
set -euo pipefail

python generate_mock_hud.py
python rl_hud_fusion.py --episodes 200 --out hud_rewards.csv
