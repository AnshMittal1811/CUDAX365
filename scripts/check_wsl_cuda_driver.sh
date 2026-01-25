#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/cuda_mobile_env.sh"

echo "# WSL CUDA Driver/Runtime Check"
echo
cuda_mobile_status
echo
echo "WSL libcuda:"
ldconfig -p 2>/dev/null | grep -E 'libcuda|libcudart' || true
echo
if [[ -f /proc/driver/nvidia/version ]]; then
  cat /proc/driver/nvidia/version
else
  echo "/proc/driver/nvidia/version unavailable; this is common in WSL2."
fi
echo
cat <<'MSG'
If CUDA programs fail with "CUDA driver version is insufficient for CUDA runtime version" in WSL2,
fix the Windows host NVIDIA driver first. Do not install a Linux NVIDIA kernel driver inside WSL.
For CUDA 12.8, install a current NVIDIA Windows driver that supports CUDA 12.8, restart Windows,
then re-open WSL and rerun this check.
MSG
