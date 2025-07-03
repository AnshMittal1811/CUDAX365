#!/usr/bin/env bash
set -euo pipefail

python generate_mock_scene.py
python overlay_hud.py
