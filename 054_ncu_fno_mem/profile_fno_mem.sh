#!/usr/bin/env bash
set -euo pipefail
ncu --metrics lts__t_sectors_hit_rate.pct,dram__bytes_read.sum \
  python ../051_fno_cufft/fno_pde.py || true
