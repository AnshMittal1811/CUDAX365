# Day 18.5: Binary Black Hole Accretion (CUDA Graphs + DynPar)

This folder simulates a synthetic accretion flow around a **binary black hole** system using CUDA
kernels (dynamic parallelism), a CUDA Graph pipeline, and cuFFT/cuBLAS/cuRAND.
The 3D render colors the **rho surface** with a **mixed color** from rho + phi (symlog phi by default).

## Governing Equations (synthetic model)

Binary BH positions (center of mass at origin):

$$
\mathbf{r}_1(t) = \frac{a}{2}[\cos(\Omega_b t),\,\sin(\Omega_b t)],\quad
\mathbf{r}_2(t) = -\mathbf{r}_1(t)
$$

Pseudo-Newtonian potential (Paczynski–Wiita):

$$
\phi(\mathbf{r}, t) = -\frac{G M_1}{|\mathbf{r}-\mathbf{r}_1(t)|-r_{s1}} -
\frac{G M_2}{|\mathbf{r}-\mathbf{r}_2(t)|-r_{s2}}
$$

Disk density with spiral + binary modulation:

$$
\rho(\mathbf{r}, t) = \rho_0\,e^{-\frac{(r-r_0)^2}{2\sigma^2}}\,[1 + A\sin(m(\theta-\Omega_d t))]\,[1 + B\cos(2\theta-\Omega_b t)]
$$

Spectral smoothing (phi):

$$
\hat{\phi}(k_x, k_y) \leftarrow \hat{\phi}(k_x, k_y)\,\exp\left(-\frac{k^2}{k_c^2}\right)
$$

Color mixing for visualization (two base colors $C_\rho$ and $C_\phi$):

$$
\text{color} = \text{norm}(\rho)\,C_\rho + \text{norm}(\phi)\,C_\phi
$$

## CUDA Pipeline

- **Dynamic parallelism:** parent kernel launches a refinement kernel near each BH.
- **CUDA Graph:** graph captures per-frame pipeline for faster replay.
- **cuFFT:** smooths the phi field in frequency space.
- **cuRAND:** generates noise perturbations.
- **cuBLAS:** adds noise into rho via SAXPY.

## Run (from this folder)

```bash
./run_binary_bh_sim.sh
```

This runs a 2-minute simulation at 12 FPS (1440 frames). To preview faster, reduce `--frames` or `--fps`.

## Notes

- Build is optimized for SM_89 by default; override with `ARCH=sm_90` or similar.
- Output frames live in `frames_binary/` and the animation is `binary_bh_3d.mp4`.
- The renderer mixes two colors to highlight peaks/troughs in rho and phi.
- Adjust color sensitivity with `--rho-gamma`, `--phi-gamma`, and `--phi-scale` in `animate_binary_bh.py`.
