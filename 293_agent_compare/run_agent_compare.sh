#!/usr/bin/env bash
set -euo pipefail

if [[ ! -f ../292_agent_configs/config_results.json ]]; then
  (cd ../292_agent_configs && ./run_agent_configs.sh)
fi

python compare_agent_manual.py
