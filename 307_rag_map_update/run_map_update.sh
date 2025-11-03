#!/usr/bin/env bash
set -euo pipefail

if [[ ! -f ../306_news_parse/locations.txt ]]; then
  (cd ../306_news_parse && ./run_news_parse.sh)
fi

python update_map_db.py
