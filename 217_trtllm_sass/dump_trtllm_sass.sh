#!/usr/bin/env bash
set -euo pipefail

ENGINE=${1:-../216_trtllm_v11/model.engine}

if [[ ! -f "$ENGINE" ]]; then
  echo "Engine not found: $ENGINE"
  exit 1
fi

if command -v cuobjdump >/dev/null 2>&1; then
  cuobjdump --dump-sass "$ENGINE" > trtllm_engine.sass.txt || true
  echo "Wrote trtllm_engine.sass.txt"
else
  echo "cuobjdump not found"
fi
