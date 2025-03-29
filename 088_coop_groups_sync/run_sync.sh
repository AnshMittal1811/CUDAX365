#!/usr/bin/env bash
set -euo pipefail
nvcc -O3 -arch=sm_89 -lineinfo coop_groups_sync.cu -o coop_groups_sync
./coop_groups_sync
