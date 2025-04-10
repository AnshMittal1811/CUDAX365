#!/usr/bin/env bash
set -euo pipefail
nvcc -O3 -arch=sm_89 -lineinfo globaltimer.cu -o globaltimer
./globaltimer
