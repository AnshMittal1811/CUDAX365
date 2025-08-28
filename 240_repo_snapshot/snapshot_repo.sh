#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd .. && pwd)
STAMP=$(date +%Y%m%d_%H%M%S)
OUT="snapshot_${STAMP}.tar.gz"

cd "$ROOT"

if command -v git >/dev/null 2>&1; then
  git status -sb > snapshot_status.txt
fi

tar -czf "$OUT" $(basename "$ROOT")

echo "Wrote $OUT"
