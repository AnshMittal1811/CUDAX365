#!/usr/bin/env bash
set -euo pipefail

ARCH=${ARCH:-sm_89}
N=${1:-16777216}
POWER_LIMIT=${POWER_LIMIT:-250}

nvcc -O3 -lineinfo -arch="$ARCH" stress_kernel.cu -o stress_kernel

if command -v nvidia-smi >/dev/null 2>&1; then
  CURRENT=$(nvidia-smi -q -d POWER | rg "Power Limit" | head -n 1 | awk '{print $4}')
  if [[ -n "$CURRENT" ]]; then
    echo "Current power limit: $CURRENT W"
    echo "Setting power limit to $POWER_LIMIT W"
    sudo nvidia-smi -pl "$POWER_LIMIT" || true
  fi

  nvidia-smi --query-gpu=timestamp,clocks.sm,power.draw --format=csv -l 1 > power_log.csv &
  LOGGER_PID=$!
  ./stress_kernel "$N"
  kill "$LOGGER_PID" || true

  if [[ -n "$CURRENT" ]]; then
    sudo nvidia-smi -pl "$CURRENT" || true
  fi
else
  echo "nvidia-smi not found; running kernel only"
  ./stress_kernel "$N"
fi
