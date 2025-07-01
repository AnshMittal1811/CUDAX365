#!/usr/bin/env bash
set -euo pipefail

if [[ ! -f ../182_gl_nv_copy_image/copy_times.csv ]]; then
  (cd ../182_gl_nv_copy_image && ./run_gl_copy.sh)
fi

python compare_copy.py
