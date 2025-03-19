#!/usr/bin/env bash
set -euo pipefail
# Requires glslangValidator with ray tracing support
G="raygen.rgen"
OUT="raygen.spv"
glslangValidator -V "$G" -o "$OUT"
