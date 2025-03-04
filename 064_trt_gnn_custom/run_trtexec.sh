#!/usr/bin/env bash
set -euo pipefail
python gnn_trt_export.py
trtexec --onnx=gnn.onnx --saveEngine=gnn.plan --verbose
