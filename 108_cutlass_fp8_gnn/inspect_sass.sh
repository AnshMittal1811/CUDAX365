#!/usr/bin/env bash
set -euo pipefail
cuobjdump --dump-sass ./cutlass_fp8_gnn | grep -E "HMMA|MMA" | head -n 40

