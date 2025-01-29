#!/usr/bin/env bash
set -euo pipefail
ROOT="${1:-$HOME/datasets/cityscapes}"

if [[ ! -d "$ROOT/leftImg8bit" ]]; then
  echo "Cityscapes not found at $ROOT"
  echo "Download from https://www.cityscapes-dataset.com and extract into $ROOT"
  exit 1
fi

echo "Cityscapes found at $ROOT"
