#!/usr/bin/env bash
set -euo pipefail

if [[ ! -f ../191_qgan_pennylane/qgan_samples.npy ]]; then
  (cd ../191_qgan_pennylane && ./run_qgan.sh)
fi

python qgan_cutensornet.py --samples ../191_qgan_pennylane/qgan_samples.npy --out cutensornet_log.txt
