#!/usr/bin/env bash
set -euo pipefail

if command -v chrt >/dev/null 2>&1; then
  chrt -f 20 python priority_tasks.py &
  PID1=$!
  chrt -f 5 python priority_tasks.py &
  PID2=$!
  wait "$PID1" "$PID2"
else
  echo "chrt not available; running without priorities"
  python priority_tasks.py
fi
