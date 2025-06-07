#!/usr/bin/env bash
set -euo pipefail

if [[ ! -f ../156_vulkan_atrous_denoise/noisy_image.npy ]]; then
  (cd ../156_vulkan_atrous_denoise && python generate_mock_image.py)
fi

python benchmark_denoise.py --image ../156_vulkan_atrous_denoise/noisy_image.npy --out denoise_timing.csv
