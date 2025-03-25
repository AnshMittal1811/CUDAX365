#!/usr/bin/env bash
set -euo pipefail
MASTER_ADDR=127.0.0.1 MASTER_PORT=29500 torchrun --nproc_per_node=2 nccl_loopback.py
