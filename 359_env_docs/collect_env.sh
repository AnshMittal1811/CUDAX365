
#!/usr/bin/env bash
set -euo pipefail

{
  echo "=== OS ==="
  uname -a
  echo "=== Python ==="
  python -V
  echo "=== Pip ==="
  python -m pip --version
  echo "=== CUDA ==="
  if command -v nvcc >/dev/null 2>&1; then
    nvcc --version
  else
    echo "nvcc not found"
  fi
  echo "=== NVIDIA-SMI ==="
  if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi
  else
    echo "nvidia-smi not found"
  fi
} > env_report.txt
echo "Wrote env_report.txt"
