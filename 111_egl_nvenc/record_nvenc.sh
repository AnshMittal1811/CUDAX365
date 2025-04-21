#!/usr/bin/env bash
set -euo pipefail
ffmpeg -y -f rawvideo -pix_fmt rgb24 -s 1920x1080 -r 30 -i frame_%04d.rgb \
  -c:v h264_nvenc -preset p4 out_nvenc.mp4

