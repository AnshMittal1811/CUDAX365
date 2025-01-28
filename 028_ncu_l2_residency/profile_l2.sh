#!/usr/bin/env bash
set -euo pipefail
nvcc -O3 -arch=sm_89 -lineinfo conv_vs_pde.cu -o conv_vs_pde

METRICS="lts__t_sectors_hit_rate.pct,lts__t_sectors_pipe_lsu_mem_global_op_ld.sum"

ncu --metrics "$METRICS" --kernel-name regex:conv3x3 ./conv_vs_pde || true
ncu --metrics "$METRICS" --kernel-name regex:pde_step ./conv_vs_pde || true
