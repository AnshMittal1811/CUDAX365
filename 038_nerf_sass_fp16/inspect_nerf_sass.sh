#!/usr/bin/env bash
set -euo pipefail
BIN="${1:-./instant-ngp/build/testbed}"
cuobjdump --dump-sass "$BIN" | grep -E "HMMA|MMA|LDMatrix" | head -n 40
