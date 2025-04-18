#!/usr/bin/env bash
set -euo pipefail
nsys profile -o pipeline_report --stats=true python pipeline_workload.py


