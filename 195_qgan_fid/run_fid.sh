#!/usr/bin/env bash
set -euo pipefail

python generate_mock_samples.py
python fid_score.py --real real_features.npy --fake fake_features.npy --out fid_score.txt
