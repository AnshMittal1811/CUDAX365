#!/usr/bin/env bash
set -euo pipefail

python generate_mock_images.py
python fp4_quantize.py --out fp4_quant_summary.txt
