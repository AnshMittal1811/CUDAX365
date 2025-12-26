
#!/usr/bin/env bash
set -euo pipefail

cat code_review_checklist.md
if command -v rg >/dev/null 2>&1; then
  rg -n "asm|ptx|__device__" ..
else
  grep -R -n "asm\|ptx\|__device__" .. || true
fi
