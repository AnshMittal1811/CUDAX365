#!/usr/bin/env bash
set -euo pipefail
# Expects raw frames from ../111_egl_nvenc
cp ../111_egl_nvenc/frame_*.rgb .

ffmpeg -y -f rawvideo -pix_fmt rgb24 -s 1920x1080 -r 30 -i frame_%04d.rgb \
  -c:v h264_nvenc -preset p4 out_nvenc.mp4 2> nvenc.log

ffmpeg -y -f rawvideo -pix_fmt rgb24 -s 1920x1080 -r 30 -i frame_%04d.rgb \
  -c:v libx264 -preset veryfast out_cpu.mp4 2> cpu.log

grep -E "fps=" nvenc.log | tail -n 1

grep -E "fps=" cpu.log | tail -n 1
