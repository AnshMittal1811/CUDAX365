#!/usr/bin/env bash
set -euo pipefail

python pipeline_orchestrator.py --nx 128 --ny 128 --steps 50 --frames 8 --out pipeline_report.json
