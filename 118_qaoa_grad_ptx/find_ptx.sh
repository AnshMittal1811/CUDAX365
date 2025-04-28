#!/usr/bin/env bash
set -euo pipefail
find torchinductor_cache -name "*.ptx" | head -n 10
