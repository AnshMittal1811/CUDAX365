#!/usr/bin/env bash
set -euo pipefail
SRC="sm90_compare.cu"
CUDA_HOME="${CUDA_HOME:-/usr/local/cuda-12.8}"
NVCC="${NVCC:-${CUDA_HOME}/bin/nvcc}"
CUOBJDUMP="${CUOBJDUMP:-${CUDA_HOME}/bin/cuobjdump}"
NATIVE_ARCH="${NATIVE_ARCH:-sm_89}"
BASELINE_ARCH="${BASELINE_ARCH:-sm_86}"
OPTIONAL_ARCH="${OPTIONAL_ARCH:-sm_90}"
OUTDIR="${OUTDIR:-build}"

if [[ ! -x "$NVCC" ]]; then
    NVCC="$(command -v nvcc || true)"
fi
if [[ ! -x "$CUOBJDUMP" ]]; then
    CUOBJDUMP="$(command -v cuobjdump || true)"
fi
if [[ -z "$NVCC" || ! -x "$NVCC" ]]; then
    echo "nvcc not found. Set CUDA_HOME or NVCC." >&2
    exit 1
fi
if [[ -z "$CUOBJDUMP" || ! -x "$CUOBJDUMP" ]]; then
    echo "cuobjdump not found. Set CUDA_HOME or CUOBJDUMP." >&2
    exit 1
fi

mkdir -p "$OUTDIR"
echo "Using NVCC=$NVCC"
echo "Using CUOBJDUMP=$CUOBJDUMP"
echo "Native RTX 4090 Laptop target: $NATIVE_ARCH"

"$NVCC" -O3 -arch="$NATIVE_ARCH" -lineinfo -Xptxas -v "$SRC" -o "$OUTDIR/saxpy_${NATIVE_ARCH}"
"$NVCC" -O3 -arch="$BASELINE_ARCH" --cubin "$SRC" -o "$OUTDIR/saxpy_${BASELINE_ARCH}.cubin"
"$NVCC" -O3 -arch="$NATIVE_ARCH" --cubin "$SRC" -o "$OUTDIR/saxpy_${NATIVE_ARCH}.cubin"

"$CUOBJDUMP" --dump-sass "$OUTDIR/saxpy_${BASELINE_ARCH}.cubin" > "$OUTDIR/sass_${BASELINE_ARCH}.txt"
"$CUOBJDUMP" --dump-sass "$OUTDIR/saxpy_${NATIVE_ARCH}.cubin" > "$OUTDIR/sass_${NATIVE_ARCH}.txt"
diff -u "$OUTDIR/sass_${BASELINE_ARCH}.txt" "$OUTDIR/sass_${NATIVE_ARCH}.txt" | sed -n '1,200p' > "$OUTDIR/sass_diff_${BASELINE_ARCH}_vs_${NATIVE_ARCH}.txt" || true

if "$NVCC" -O3 -arch="$OPTIONAL_ARCH" --cubin "$SRC" -o "$OUTDIR/saxpy_${OPTIONAL_ARCH}.cubin"; then
    "$CUOBJDUMP" --dump-sass "$OUTDIR/saxpy_${OPTIONAL_ARCH}.cubin" > "$OUTDIR/sass_${OPTIONAL_ARCH}.txt"
    diff -u "$OUTDIR/sass_${NATIVE_ARCH}.txt" "$OUTDIR/sass_${OPTIONAL_ARCH}.txt" | sed -n '1,200p' > "$OUTDIR/sass_diff_${NATIVE_ARCH}_vs_${OPTIONAL_ARCH}.txt" || true
else
    echo "Optional $OPTIONAL_ARCH cubin compile skipped by this CUDA/toolchain." > "$OUTDIR/sass_diff_${NATIVE_ARCH}_vs_${OPTIONAL_ARCH}.txt"
fi

if "$OUTDIR/saxpy_${NATIVE_ARCH}" > "$OUTDIR/run_${NATIVE_ARCH}.txt" 2>&1; then
    cat "$OUTDIR/run_${NATIVE_ARCH}.txt"
else
    echo "Runtime execution failed; compile and SASS dump still completed." >> "$OUTDIR/run_${NATIVE_ARCH}.txt"
    cat "$OUTDIR/run_${NATIVE_ARCH}.txt"
fi

echo "Wrote:"
echo "  $OUTDIR/sass_${BASELINE_ARCH}.txt"
echo "  $OUTDIR/sass_${NATIVE_ARCH}.txt"
if [[ -f "$OUTDIR/sass_${OPTIONAL_ARCH}.txt" ]]; then
    echo "  $OUTDIR/sass_${OPTIONAL_ARCH}.txt"
fi
echo "  $OUTDIR/sass_diff_${BASELINE_ARCH}_vs_${NATIVE_ARCH}.txt"
echo "  $OUTDIR/sass_diff_${NATIVE_ARCH}_vs_${OPTIONAL_ARCH}.txt"
