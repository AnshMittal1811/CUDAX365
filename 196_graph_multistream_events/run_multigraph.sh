#!/usr/bin/env bash
set -euo pipefail

ARCH=${ARCH:-sm_89}
N=${1:-1048576}

nvcc -O3 -lineinfo -arch="$ARCH" multigraph_events.cu -o multigraph_events
./multigraph_events "$N"
