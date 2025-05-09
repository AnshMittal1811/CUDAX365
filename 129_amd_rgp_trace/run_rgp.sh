#!/usr/bin/env bash
set -euo pipefail
if ! command -v rgp >/dev/null 2>&1; then
  echo "RGP not found. Install Radeon GPU Profiler and ensure rgp is in PATH."
  exit 1
fi
rgp --capture ./some_gpu_app
