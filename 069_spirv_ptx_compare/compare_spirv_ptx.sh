#!/usr/bin/env bash
set -euo pipefail
# Placeholder: run Vulkan and CUDA versions and compare timings
python - <<'PY'
import time
import numpy as np
n = 1 << 20
a = np.random.randn(n).astype(np.float32)
start = time.time()
for _ in range(100):
    a = a * 0.99
end = time.time()
print("cpu_baseline_ms", (end-start)*1000)
PY
