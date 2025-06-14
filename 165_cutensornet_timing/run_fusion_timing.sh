#!/usr/bin/env bash
set -euo pipefail

if [[ ! -f ../164_cutensornet_fusion/fusion_metrics.txt ]]; then
  (cd ../164_cutensornet_fusion && ./run_fusion.sh)
fi

python log_fusion_times.py --metrics ../164_cutensornet_fusion/fusion_metrics.txt --out fusion_times.csv
