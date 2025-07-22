#!/usr/bin/env bash
set -euo pipefail

python generate_mock_inputs.py
python pipeline_int4.py
