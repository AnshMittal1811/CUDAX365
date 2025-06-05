#!/usr/bin/env bash
set -euo pipefail

python generate_mock_image.py
python atrous_cpu.py

if command -v glslangValidator >/dev/null 2>&1; then
  glslangValidator -V atrous_denoise.comp -o atrous_denoise.spv
  echo "Compiled atrous_denoise.comp to atrous_denoise.spv"
else
  echo "glslangValidator not found; skipped shader compilation"
fi
