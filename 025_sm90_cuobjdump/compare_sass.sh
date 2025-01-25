#!/usr/bin/env bash
set -euo pipefail
SRC="sm90_compare.cu"

nvcc -O3 -arch=sm_86 "$SRC" -o sm86
nvcc -O3 -arch=sm_90 "$SRC" -o sm90

cuobjdump --dump-sass sm86 > sass_sm86.txt
cuobjdump --dump-sass sm90 > sass_sm90.txt

diff -u sass_sm86.txt sass_sm90.txt | head -n 200 > sass_diff.txt || true

echo "Wrote sass_sm86.txt, sass_sm90.txt, sass_diff.txt"
