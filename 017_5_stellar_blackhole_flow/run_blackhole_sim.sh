
#!/usr/bin/env bash
set -euo pipefail

# 2-minute simulation @12 FPS => 1440 frames
# Reduce --frames or --fps for a faster preview.
if command -v nvcc >/dev/null 2>&1; then
  ./build_blackhole_cuda.sh
  ./blackhole_flow_cuda --nx 160 --ny 160 --frames 1440 --dt 0.05 --out-hot frames_hot --out-cold frames_cold
else
  python generate_blackhole_frames.py --nx 160 --ny 160 --frames 1440 --dt 0.05
fi
python animate_blackhole_flow.py --shape 160 160 --stride 2 --fps 12 --phi-scale symlog --out blackhole_flow_3d.mp4
