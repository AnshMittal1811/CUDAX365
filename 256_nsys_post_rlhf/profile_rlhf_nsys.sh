#!/usr/bin/env bash
set -euo pipefail

if ! command -v nsys >/dev/null 2>&1; then
  echo "nsys not found"
  exit 1
fi

if [[ ! -f ../255_rlhf_integration/rlhf_integration_results.csv ]]; then
  (cd ../255_rlhf_integration && ./run_rlhf_integration.sh)
fi

nsys profile -o rlhf_pipeline_report --force-overwrite true --sample=none --trace=cuda,nvtx \
  bash -c "cd ../255_rlhf_integration && python integrate_rlhf.py"
