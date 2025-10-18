#!/usr/bin/env bash
set -euo pipefail

if [[ ! -f ../291_autogpt_tuner/autotune_proposals.json ]]; then
  (cd ../291_autogpt_tuner && ./run_auto_tune.sh)
fi

python evaluate_configs.py
