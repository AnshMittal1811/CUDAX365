
#!/usr/bin/env bash
set -euo pipefail

if command -v nvidia-cuda-mps-control >/dev/null 2>&1; then
  echo "Starting MPS control..."
  nvidia-cuda-mps-control -d
  python ../343_concurrent_inference/concurrent_inference.py
  echo "quit" | nvidia-cuda-mps-control
else
  echo "MPS not available; running baseline concurrent inference."
  python ../343_concurrent_inference/concurrent_inference.py
fi
