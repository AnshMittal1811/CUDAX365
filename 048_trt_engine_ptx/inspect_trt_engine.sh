#!/usr/bin/env bash
set -euo pipefail
ENGINE="${1:-./trtllm_engine/engine.plan}"
if [[ ! -f "$ENGINE" ]]; then
  echo "engine plan not found: $ENGINE"
  exit 1
fi
strings "$ENGINE" | grep -E "HMMA|DP4A" | head -n 20
