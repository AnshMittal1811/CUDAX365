#!/usr/bin/env bash
set -euo pipefail

if ! command -v git >/dev/null 2>&1; then
  echo "git not found"
  exit 1
fi

git status -sb

echo "Review data for sensitive content before pushing."

echo "Suggested steps:"
cat <<'STEPS'
1) git remote add origin https://github.com/<user>/<repo>.git
2) git push -u origin main
STEPS
