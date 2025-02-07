#!/usr/bin/env bash
set -euo pipefail
ROOT="${1:-./instant-ngp}"
mkdir -p "$ROOT/build_fp16"
cd "$ROOT/build_fp16"
cmake .. -DTCNN_HALF=ON
cmake --build . -j
