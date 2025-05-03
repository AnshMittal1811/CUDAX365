#!/usr/bin/env bash
set -euo pipefail
# Requires Nsight Compute GUI to view occupancy heatmap
ncu --section Occupancy --target-processes all ../122_fp8_triplebuf/fp8_triplebuf || true
