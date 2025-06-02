#!/usr/bin/env bash
set -euo pipefail

if [[ ! -f ../151_curand_sobol_quality/sobol_device.bin || ! -f ../151_curand_sobol_quality/sobol_host.bin ]]; then
  echo "Running Day 151 generator..."
  (cd ../151_curand_sobol_quality && ./run_sobol_compare.sh)
fi

if [[ ! -f ../152_ptx_xorshift_warp/xorshift.bin ]]; then
  echo "Running Day 152 generator..."
  (cd ../152_ptx_xorshift_warp && ./run_xorshift.sh)
fi

python rng_quality_compare.py --out rng_quality_report.json
