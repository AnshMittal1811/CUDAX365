#!/usr/bin/env bash
set -euo pipefail
nvcc -O3 -arch=sm_89 -lineinfo nerf_sampling.cu -o nerf_sampling
ncu --metrics lts__t_sectors_hit_rate.pct,tex__t_sectors_hit_rate.pct ./nerf_sampling || true
