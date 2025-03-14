#!/usr/bin/env bash
set -euo pipefail
deepspeed --version >/dev/null 2>&1 || true
python train_tinyllama_ds.py
