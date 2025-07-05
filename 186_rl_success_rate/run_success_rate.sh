#!/usr/bin/env bash
set -euo pipefail

if [[ ! -f ../154_rl_random_resets/rewards_random_resets.csv ]]; then
  (cd ../154_rl_random_resets && ./run_random_resets.sh)
fi

if [[ ! -f ../185_rl_hud_fusion/hud_rewards.csv ]]; then
  (cd ../185_rl_hud_fusion && ./run_hud_fusion.sh)
fi

python compare_success_rate.py
