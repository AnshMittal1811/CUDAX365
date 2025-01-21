#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <kernel.cubin> [out.sass]"
  exit 1
fi

IN="$1"
OUT="${2:-kernel.sass}"

nvdisasm --print-code --print-line-info "$IN" > "$OUT"
python annotate_sass.py "$OUT" "${OUT%.sass}.annotated.sass"
echo "Wrote ${OUT%.sass}.annotated.sass"
