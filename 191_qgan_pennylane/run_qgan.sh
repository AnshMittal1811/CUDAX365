#!/usr/bin/env bash
set -euo pipefail

python generate_toy_data.py
python qgan_pennylane.py --real real_samples.npy --out qgan_samples.npy
