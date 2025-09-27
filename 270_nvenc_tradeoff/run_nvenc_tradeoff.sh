#!/usr/bin/env bash
set -euo pipefail

if [[ ! -f ../269_nvenc_preset_tuning/nvenc_outputs.csv ]]; then
  (cd ../269_nvenc_preset_tuning && ./run_nvenc_presets.sh)
fi

python plot_nvenc_tradeoff.py
