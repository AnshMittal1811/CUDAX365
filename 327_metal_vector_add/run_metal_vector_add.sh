
#!/usr/bin/env bash
set -euo pipefail

if [[ "$(uname)" == "Darwin" ]] && command -v xcrun >/dev/null 2>&1; then
  echo "Compiling Metal shader..."
  xcrun -sdk macosx metal -c vector_add.metal -o vector_add.air
  xcrun -sdk macosx metallib vector_add.air -o vector_add.metallib
  echo "Compiled vector_add.metallib (run via a Metal host app)."
else
  echo "macOS Metal toolchain not available; running CPU fallback."
  python vector_add_stub.py
fi
