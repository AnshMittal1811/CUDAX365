#!/usr/bin/env bash
set -euo pipefail

TMP_DIRS=(/tmp ./tmp)

for dir in "${TMP_DIRS[@]}"; do
  if [[ -d "$dir" ]]; then
    echo "Cleaning $dir"
    find "$dir" -maxdepth 1 -type f -mtime +7 -print -delete || true
  fi
done

if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi -L > gpu_info.txt
fi

if command -v nvcc >/dev/null 2>&1; then
  nvcc --version > cuda_version.txt
fi

echo "Maintenance done"
