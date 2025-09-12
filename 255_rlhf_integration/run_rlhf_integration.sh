#!/usr/bin/env bash
set -euo pipefail

if [[ ! -f ../254_rlhf_policy_train/rlhf_train_log.txt ]]; then
  (cd ../254_rlhf_policy_train && ./run_rlhf_train.sh)
fi

python integrate_rlhf.py
