#!/usr/bin/env bash
set -euo pipefail
python export_onnx.py
python trt10_fp4.py
