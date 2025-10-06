#!/usr/bin/env bash
set -euo pipefail

if [[ ! -f ../277_mixed_precision_conv/mixed_precision_log.txt ]]; then
  (cd ../277_mixed_precision_conv && ./run_mixed_precision.sh)
fi

python compare_throughput.py
