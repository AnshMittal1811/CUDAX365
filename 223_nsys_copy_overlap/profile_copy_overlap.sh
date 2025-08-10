#!/usr/bin/env bash
set -euo pipefail

if ! command -v nsys >/dev/null 2>&1; then
  echo "nsys not found"
  exit 1
fi

(cd ../221_copy_engine_saturation && ./run_copy_saturation.sh 4194304 4)

nsys profile -o copy_overlap_report --force-overwrite true --sample=none --trace=cuda ../221_copy_engine_saturation/copy_saturation 4194304 4
