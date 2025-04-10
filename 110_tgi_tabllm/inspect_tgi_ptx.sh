#!/usr/bin/env bash
set -euo pipefail
CACHE=${TRITON_CACHE_DIR:-$HOME/.triton}
find "$CACHE" -name "*.ptx" | head -n 20
