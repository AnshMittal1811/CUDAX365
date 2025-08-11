#!/usr/bin/env bash
set -euo pipefail

python chunk_copy_bench.py --size 8000000 --out chunk_copy.csv
