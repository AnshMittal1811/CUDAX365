#!/usr/bin/env bash
set -euo pipefail

ROOT=${1:-./torchinductor_cache}

if [[ ! -d "$ROOT" ]]; then
  echo "TorchInductor cache not found: $ROOT"
  exit 1
fi

rg -n "\.ptx" "$ROOT" || true
