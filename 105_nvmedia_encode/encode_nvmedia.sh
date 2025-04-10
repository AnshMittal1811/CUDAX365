#!/usr/bin/env bash
set -euo pipefail

# Try NvMedia if SDK is available, otherwise fallback to ffmpeg NVENC
if [[ -n "${NVMEDIA_ROOT:-}" ]] && [[ -x "$NVMEDIA_ROOT/samples/nvm_encode" ]]; then
  echo "Using NvMedia encoder"
  "$NVMEDIA_ROOT/samples/nvm_encode" || true
else
  echo "NvMedia not found; using ffmpeg h264_nvenc if available"
  ffmpeg -y -f rawvideo -pix_fmt rgb24 -s 320x180 -r 30 -i frame_%04d.rgb \
    -c:v h264_nvenc -preset p4 out_nvenc.mp4 || \
  ffmpeg -y -f rawvideo -pix_fmt rgb24 -s 320x180 -r 30 -i frame_%04d.rgb \
    -c:v libx264 -preset veryfast out_cpu.mp4
fi
