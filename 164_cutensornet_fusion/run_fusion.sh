#!/usr/bin/env bash
set -euo pipefail

python generate_mock_tensor.py
python cutensornet_fusion.py --input input_tensor.npy --out fusion_metrics.txt
