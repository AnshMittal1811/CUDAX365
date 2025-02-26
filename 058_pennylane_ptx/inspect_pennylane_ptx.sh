#!/usr/bin/env bash
set -euo pipefail
LIB=$(python - <<'PY'
import pennylane_lightning_gpu as pl, inspect, os
print(os.path.dirname(inspect.getfile(pl)))
PY
)
SO=$(find "$LIB" -name "*.so" | head -n 1)
cuobjdump --dump-ptx "$SO" | head -n 40
