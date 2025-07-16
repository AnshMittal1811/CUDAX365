#!/usr/bin/env bash
set -euo pipefail

if ! command -v nsys >/dev/null 2>&1; then
  echo "nsys not found"
  exit 1
fi

(cd ../196_graph_multistream_events && ./run_multigraph.sh 1048576)

nsys profile -o overlap_events_report --force-overwrite true --sample=none --trace=cuda,nvtx ../196_graph_multistream_events/multigraph_events 1048576
