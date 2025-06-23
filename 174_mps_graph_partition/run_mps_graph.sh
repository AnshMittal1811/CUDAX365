#!/usr/bin/env bash
set -euo pipefail

ARCH=${ARCH:-sm_89}
N=${1:-1048576}
ITERS=${2:-100}

nvcc -O3 -lineinfo -arch="$ARCH" graph_worker.cu -o graph_worker

if [[ -z "${CUDA_MPS_PIPE_DIRECTORY:-}" ]]; then
  export CUDA_MPS_PIPE_DIRECTORY=/tmp/nvidia-mps
  export CUDA_MPS_LOG_DIRECTORY=/tmp/nvidia-mps
fi

if command -v nvidia-cuda-mps-control >/dev/null 2>&1; then
  echo "Starting MPS"
  nvidia-cuda-mps-control -d
  ./graph_worker "$N" "$ITERS" &
  PID1=$!
  ./graph_worker "$N" "$ITERS" &
  PID2=$!
  wait "$PID1" "$PID2"
  echo quit | nvidia-cuda-mps-control
else
  echo "MPS control not found; running two processes without MPS"
  ./graph_worker "$N" "$ITERS" &
  PID1=$!
  ./graph_worker "$N" "$ITERS" &
  PID2=$!
  wait "$PID1" "$PID2"
fi
