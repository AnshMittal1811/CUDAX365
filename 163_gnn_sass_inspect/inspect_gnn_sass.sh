#!/usr/bin/env bash
set -euo pipefail

CACHE=${1:-../162_gnn_inductor_ptx/torchinductor_cache}

if [[ ! -d "$CACHE" ]]; then
  echo "Cache not found: $CACHE"
  exit 1
fi

if ! command -v cuobjdump >/dev/null 2>&1; then
  echo "cuobjdump not found"
  exit 1
fi

find "$CACHE" -name "*.so" -o -name "*.cubin" > targets.txt

if [[ ! -s targets.txt ]]; then
  echo "No .so/.cubin found in cache"
  exit 1
fi

while read -r bin; do
  echo "Dumping SASS for $bin"
  cuobjdump --dump-sass "$bin" > "$(basename "$bin").sass.txt" || true
  break
done < targets.txt
