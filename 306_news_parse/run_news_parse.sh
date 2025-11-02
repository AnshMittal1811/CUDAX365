#!/usr/bin/env bash
set -euo pipefail

python generate_news.py
python extract_locations.py
