#!/usr/bin/env bash
set -euo pipefail
nsys profile -o transfer_report --stats=true python transfer_overlap.py
