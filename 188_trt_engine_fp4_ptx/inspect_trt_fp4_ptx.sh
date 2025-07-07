#!/usr/bin/env bash
set -euo pipefail

if ! command -v trtexec >/dev/null 2>&1; then
  echo "trtexec not found"
  exit 1
fi

trtexec --onnx=mock.onnx --saveEngine=fp4.engine --fp4 2>/dev/null || true

if command -v cuobjdump >/dev/null 2>&1; then
  cuobjdump --dump-ptx fp4.engine > fp4_engine.ptx.txt || true
  echo "Wrote fp4_engine.ptx.txt"
else
  echo "cuobjdump not found"
fi
