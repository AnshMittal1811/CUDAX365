#!/usr/bin/env bash
set -euo pipefail

if ! command -v ncu >/dev/null 2>&1; then
  echo "ncu not found"
  exit 1
fi

# Use a lightweight kernel from Day 228
if [[ ! -f ../228_epilogue_speed_compare/epilogue_bench ]]; then
  (cd ../228_epilogue_speed_compare && ./run_epilogue_bench.sh 128 128)
fi

ncu --section InstructionStats --section SourceCounters --target-processes all \
  ../228_epilogue_speed_compare/epilogue_bench 128 128 | tee instruction_timeline.txt
