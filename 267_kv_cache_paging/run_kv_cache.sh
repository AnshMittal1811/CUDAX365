#!/usr/bin/env bash
set -euo pipefail

python kv_cache_paging.py --size 1000000 --cache kv_cache.dat
