#!/usr/bin/env bash
set -euo pipefail

if [[ ! -f ../191_qgan_pennylane/qgan_samples.npy ]]; then
  (cd ../191_qgan_pennylane && ./run_qgan.sh)
fi

python qgan_rl_domain.py --samples ../191_qgan_pennylane/qgan_samples.npy --out qgan_rl_results.csv
