#!/usr/bin/env bash
set -euo pipefail
LIB=$(ldconfig -p | grep libnccl.so | head -n1 | awk '{print $NF}')
if [[ -z "$LIB" ]]; then
  echo "libnccl.so not found"
  exit 1
fi
cuobjdump --dump-sass "$LIB" | grep -E "HMMA|MMA" | head -n 40
