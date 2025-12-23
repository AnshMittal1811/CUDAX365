
#!/usr/bin/env bash
set -euo pipefail

if command -v rg >/dev/null 2>&1; then
  rg -n "TODO|FIXME" ..
else
  grep -R -n "TODO\|FIXME" .. || true
fi
