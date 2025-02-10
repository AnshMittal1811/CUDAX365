#!/usr/bin/env bash
set -euo pipefail
python prepare_mhd_logs.py
python qlora_tinyllama.py --steps 20
