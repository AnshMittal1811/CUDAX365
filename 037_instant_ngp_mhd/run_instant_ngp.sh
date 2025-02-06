#!/usr/bin/env bash
set -euo pipefail
python generate_mhd_volume.py

if [[ ! -d instant-ngp ]]; then
  git clone --recursive https://github.com/NVlabs/instant-ngp.git
fi

cd instant-ngp
mkdir -p build
cd build
cmake ..
cmake --build . -j

./testbed --scene ../mhd_volume/volume.json
