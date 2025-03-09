#!/usr/bin/env bash
set -euo pipefail
# Requires glslangValidator
G="pde_comp.comp"
OUT="pde_comp.spv"
glslangValidator -V "$G" -o "$OUT"
