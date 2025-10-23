#!/usr/bin/env bash
set -euo pipefail

if ! command -v nsys >/dev/null 2>&1; then
  echo "nsys not found"
  exit 1
fi

nsys profile -o integration_12h_report --force-overwrite true --sample=none --trace=cuda \
  bash -c "cd ../296_integration_test_12h && ./run_integration_12h.sh 12"
