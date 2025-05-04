#!/usr/bin/env bash
set -euo pipefail
nvcc -O3 -arch=sm_89 -lineinfo coop_multistream.cu -o coop_multistream
./coop_multistream
