#!/usr/bin/env bash
set -euo pipefail
python export_onnx.py
trtexec --onnx=mlp.onnx --int8 --saveEngine=mlp_int8.plan
