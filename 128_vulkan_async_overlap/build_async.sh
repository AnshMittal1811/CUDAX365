#!/usr/bin/env bash
set -euo pipefail
# Requires glslangValidator
for f in async_compute.comp async_render.frag; do
  glslangValidator -V "$f" -o "${f}.spv"
done
