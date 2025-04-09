#!/usr/bin/env bash
set -euo pipefail
A=$1
B=$2
python count_ptx.py "$A"
python count_ptx.py "$B"
