#!/usr/bin/env bash
set -euo pipefail

ARCH=${ARCH:-sm_89}
N=${N:-16777216}

nvcc -O3 -lineinfo -arch="$ARCH" stress_kernel.cu -o stress_kernel

if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "nvidia-smi not found; running kernel once"
  ./stress_kernel "$N"
  exit 0
fi

CURRENT=$(nvidia-smi -q -d POWER | rg "Power Limit" | head -n 1 | awk '{print $4}')
LIMITS=(200 220 240 260 280)

printf "power_limit,elapsed_ms\n" > thermal_sweep.csv

for limit in "${LIMITS[@]}"; do
  echo "Setting power limit to $limit W"
  sudo nvidia-smi -pl "$limit" || true
  start=$(date +%s%3N)
  ./stress_kernel "$N" >/dev/null
  end=$(date +%s%3N)
  elapsed=$((end - start))
  printf "%s,%s\n" "$limit" "$elapsed" >> thermal_sweep.csv
  sleep 1
  nvidia-smi --query-gpu=temperature.gpu --format=csv -l 1 -c 1 >> thermal_temps.csv
  sleep 1
done

if [[ -n "$CURRENT" ]]; then
  sudo nvidia-smi -pl "$CURRENT" || true
fi

echo "Wrote thermal_sweep.csv and thermal_temps.csv"
