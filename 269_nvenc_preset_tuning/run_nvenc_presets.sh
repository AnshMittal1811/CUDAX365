#!/usr/bin/env bash
set -euo pipefail

python generate_frames.py

PRESETS=(p1 p2 p3 p4)

: > nvenc_outputs.csv

for preset in "${PRESETS[@]}"; do
  OUT="output_${preset}.mp4"
  if command -v ffmpeg >/dev/null 2>&1; then
    ffmpeg -y -f rawvideo -pixel_format rgb24 -video_size 256x256 -framerate 30 \
      -i frame_%03d.raw -c:v h264_nvenc -preset "$preset" "$OUT" 2> "nvenc_${preset}.log" || true
  else
    echo "ffmpeg not found" > "nvenc_${preset}.log"
  fi
  echo "$preset,$OUT" >> nvenc_outputs.csv
done
