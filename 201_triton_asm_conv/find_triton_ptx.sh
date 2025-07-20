#!/usr/bin/env bash
set -euo pipefail

if [[ ! -d ./triton_dump ]]; then
  echo "triton_dump not found; run triton_asm_conv.py first"
  exit 1
fi

rg -n "ptx" ./triton_dump || true
