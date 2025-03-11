#!/usr/bin/env bash
set -euo pipefail
g++ -O2 egl_headless.cpp -lEGL -lGLESv2 -o egl_headless
./egl_headless
