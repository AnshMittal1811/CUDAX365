#!/usr/bin/env bash
set -euo pipefail
DURATION_SEC=$((6*60*60))
nsys profile -o stress_report --stats=true \
  bash -c "python - <<'PY'\nimport time\nend = time.time() + $DURATION_SEC\nwhile time.time() < end:\n    time.sleep(1)\nprint('stress done')\nPY"
