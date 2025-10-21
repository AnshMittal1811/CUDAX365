#!/usr/bin/env bash
set -euo pipefail

python generate_papers.py
python extract_citations.py
