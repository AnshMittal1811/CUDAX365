#!/usr/bin/env bash
set -euo pipefail

# Shared RTX 4090 Laptop / WSL2 CUDA defaults for the 250 day CUDA exercises.
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda-12.8}"
if [[ -x "$CUDA_HOME/bin/nvcc" ]]; then
  export PATH="$CUDA_HOME/bin:$PATH"
fi
if [[ -d "$CUDA_HOME/lib64" ]]; then
  export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"
fi
if [[ -d "$CUDA_HOME/targets/x86_64-linux/lib" ]]; then
  export LD_LIBRARY_PATH="$CUDA_HOME/targets/x86_64-linux/lib:${LD_LIBRARY_PATH:-}"
fi

export NVCC="${NVCC:-$(command -v nvcc || true)}"
export CUOBJDUMP="${CUOBJDUMP:-$(command -v cuobjdump || true)}"
export NATIVE_ARCH="${NATIVE_ARCH:-sm_89}"
export ARCH="${ARCH:-$NATIVE_ARCH}"
export BASELINE_ARCH="${BASELINE_ARCH:-sm_86}"
export OPTIONAL_ARCH="${OPTIONAL_ARCH:-sm_90}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.9}"
export NVIDIA_TF32_OVERRIDE="${NVIDIA_TF32_OVERRIDE:-0}"

cuda_mobile_status() {
  echo "CUDA_HOME=${CUDA_HOME}"
  echo "NVCC=${NVCC:-missing}"
  echo "CUOBJDUMP=${CUOBJDUMP:-missing}"
  echo "ARCH=${ARCH}"
  echo "NATIVE_ARCH=${NATIVE_ARCH}"
  echo "TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST}"
  if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi --query-gpu=name,compute_cap,memory.total,driver_version --format=csv,noheader || true
  else
    echo "nvidia-smi=missing"
  fi
  if [[ -n "${NVCC:-}" && -x "${NVCC:-}" ]]; then
    "$NVCC" --version | tail -n 1 || true
  fi
}
