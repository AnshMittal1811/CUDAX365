# 014_mhd_fp16_tensorcore

Goal:
- Run an FP16, WMMA-based MHD-like update with a Poisson solve for phi, and verify HMMA in SASS/PTX.


Build:
```bash
nvcc -O3 -arch=sm_89 -lineinfo -Xptxas -v -std=c++17 mhd_fp16_tc.cu -lcufft -o mhd_fp16_tc
```

Run (NX NY STEPS mix phi_scale):
```bash
./mhd_fp16_tc 128 128 60
```

Verify Tensor Core instructions:
```bash
nvcc -O3 -arch=sm_89 -lineinfo -Xptxas -v -std=c++17 -ptx mhd_fp16_tc.cu -o mhd_fp16_tc.ptx
rg -n "mma.sync" mhd_fp16_tc.ptx
cuobjdump --dump-sass mhd_fp16_tc | rg "HMMA|mma"
```

Animate rho + phi in a single frame:
```bash
python animate_mhd_fp16.py --rho "frames/rho_*.bin" --phi "frames/phi_*.bin" \
  --shape 128 128 --out mhd_rho_phi.mp4 --fps 12 --title "FP16 MHD (rho | phi)"
```
If `ffmpeg` is missing, use `--out mhd_rho_phi.gif`.

Notes:
- NX and NY must be multiples of 16 for WMMA tiles.
- Each saved frame corresponds to one step (step size = 1).
- The WMMA update is a tile-local blur to exercise Tensor Cores; phi couples back into rho.
