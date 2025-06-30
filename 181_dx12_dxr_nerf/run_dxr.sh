#!/usr/bin/env bash
set -euo pipefail

python mock_nerf_render.py

if command -v powershell.exe >/dev/null 2>&1; then
  powershell.exe -ExecutionPolicy Bypass -File ./compile_dxr.ps1 || true
else
  echo "powershell.exe not found; skipped DXR compile"
fi
