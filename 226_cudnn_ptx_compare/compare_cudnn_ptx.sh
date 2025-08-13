#!/usr/bin/env bash
set -euo pipefail

if command -v cuobjdump >/dev/null 2>&1; then
  if [[ -f ../227_inline_ptx_activation/inline_activation ]]; then
    cuobjdump --dump-ptx ../227_inline_ptx_activation/inline_activation > manual_activation.ptx.txt || true
  fi
  echo "Saved manual_activation.ptx.txt"
else
  echo "cuobjdump not found"
fi
