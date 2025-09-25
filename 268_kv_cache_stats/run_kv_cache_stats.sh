#!/usr/bin/env bash
set -euo pipefail

if [[ ! -f ../267_kv_cache_paging/kv_cache.dat ]]; then
  (cd ../267_kv_cache_paging && ./run_kv_cache.sh)
fi

python kv_cache_stats.py --cache ../267_kv_cache_paging/kv_cache.dat --out kv_cache_stats.txt
