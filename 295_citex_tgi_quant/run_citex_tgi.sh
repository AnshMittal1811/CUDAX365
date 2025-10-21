#!/usr/bin/env bash
set -euo pipefail

python quantize_citex.py

if command -v docker >/dev/null 2>&1; then
  echo "Launch TGI with quantized model (placeholder)" > tgi_run.txt
else
  echo "docker not found" > tgi_run.txt
fi
