
#!/usr/bin/env bash
set -euo pipefail

if command -v ncu >/dev/null 2>&1; then
  ncu --target-processes all --set full python ../320_triton_int4_opt/triton_int4_opt.py --iters 20
else
  echo "ncu not found. Install Nsight Compute and re-run." >&2
  python ../320_triton_int4_opt/triton_int4_opt.py --iters 5
fi
