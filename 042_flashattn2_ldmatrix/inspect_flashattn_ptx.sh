#!/usr/bin/env bash
set -euo pipefail
LIB=$(python - <<'PY'
import flash_attn, inspect, os
print(os.path.dirname(inspect.getfile(flash_attn)))
PY
)
SO=$(find "$LIB" -name "*.so" | head -n 1)
cuobjdump --dump-ptx "$SO" | grep -E "ldmatrix" | head -n 20
