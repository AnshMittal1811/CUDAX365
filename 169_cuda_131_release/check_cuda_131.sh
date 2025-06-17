#!/usr/bin/env bash
set -euo pipefail

if command -v nvcc >/dev/null 2>&1; then
  nvcc --version | tee cuda_version.txt
else
  echo "nvcc not found" | tee cuda_version.txt
fi

if command -v curl >/dev/null 2>&1; then
  curl -L -o cuda_131_release_notes.html https://developer.nvidia.com/cuda-toolkit-archive || true
else
  echo "curl not found; skipping download" > cuda_131_release_notes.html
fi

