#!/usr/bin/env bash
set -euo pipefail
LIB="$(ldconfig -p | grep libnvinfer.so | head -n1 | awk '{print $NF}')"
if [[ -z "$LIB" ]]; then
  echo "libnvinfer.so not found"
  exit 1
fi
cuobjdump --dump-ptx "$LIB" | grep -E "HMMA|DP4A" | head -n 20
