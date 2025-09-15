#!/usr/bin/env bash
set -euo pipefail

LIMIT=${1:-220}
DURATION=${2:-60}

if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "nvidia-smi not found"
  exit 1
fi

CURRENT=$(nvidia-smi -q -d POWER | rg "Power Limit" | head -n 1 | awk '{print $4}')

sudo nvidia-smi -pl "$LIMIT" || true
nvidia-smi --query-gpu=timestamp,clocks.sm,power.draw --format=csv -l 1 -c "$DURATION" > power_limit_log.csv

if [[ -n "$CURRENT" ]]; then
  sudo nvidia-smi -pl "$CURRENT" || true
fi

