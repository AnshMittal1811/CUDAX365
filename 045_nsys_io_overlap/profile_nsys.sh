#!/usr/bin/env bash
set -euo pipefail
nsys profile -o io_overlap_report --stats=true python io_compute_overlap.py
