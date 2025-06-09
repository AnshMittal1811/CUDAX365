#!/usr/bin/env bash
set -euo pipefail

if [[ ! -f ../158_denoise_timing/denoised_image.npy ]]; then
  (cd ../158_denoise_timing && ./run_denoise_timing.sh)
fi

python compare_quality.py --clean ../156_vulkan_atrous_denoise/noisy_image.npy --denoised ../158_denoise_timing/denoised_image.npy --out quality_metrics.txt
