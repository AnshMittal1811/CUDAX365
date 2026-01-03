# Day 19.5: 8 Black Holes + SMBH Accretion (Final-Parsec Stall)

This folder simulates **8 stellar-mass black holes** (1 binary + 6 singles) orbiting a
**super-massive black hole (SMBH)** with hot/cold accretion flows. The animation shows:

- **Top panel:** combined hot + cold flow.
- **Bottom panels:** hot-only and cold-only views for the full 8-BH system.

Color encodes a **mixture of rho and phi**, while **x, y, z show the 3D motion** of the flows
(rho/phi do not drive the z axis).
A lensing-style brightness boost is applied per BH, along with a relativistic brightness factor
(gravitational redshift + Doppler). The renderer also overlays simple magnetic field loops.

## Governing Equations (synthetic model)

Binary-like orbit around SMBH with a **final-parsec stall** (semi-major axis asymptotes):

$$
 a(t) = a_{\min} + \frac{a_0 - a_{\min}}{1 + t/\tau}
$$

Orbital phase and radius:

$$
 r(t) = \frac{a(t)(1-e^2)}{1 + e\cos\theta(t)},\quad \theta(t) = \Omega_0 t + \theta_0
$$

Pseudo-Newtonian gravitational potential (Paczynski–Wiita):

$$
 \phi(\mathbf{r}) = -\frac{G M_{\mathrm{SMBH}}}{|\mathbf{r}|-r_s} - \frac{G M_{\mathrm{BH}}}{|\mathbf{r}-\mathbf{r}_i|-r_{s,\mathrm{BH}}}
$$

Hot/cold density profiles with spiral modulation:

$$
 \rho(\mathbf{r}, t) = \rho_0\,e^{-\frac{(r-r_0)^2}{2\sigma^2}}\left[1 + A\sin(m(\theta-\Omega_d t))\right]
$$

Relativistic brightness (approximate):

$$
 I \propto \frac{\sqrt{1-r_s/r}}{\gamma(1-\beta\cos\mu)}
$$

Lensing-style magnification (heuristic):

$$
 I \leftarrow I\left[1 + \\alpha\\left(\\frac{r_{s}^2}{r^2 + \\epsilon} + \\frac{r_{s,\\mathrm{BH}}^2}{r_{\\mathrm{BH}}^2 + \\epsilon}\\right)\\right]
$$

Color mixing for visualization (two base colors $C_\rho$ and $C_\phi$):

$$
 \text{color} = \text{norm}(\rho)\,C_\rho + \text{norm}(\phi)\,C_\phi
$$

## Run (from this folder)

```bash
./run_multi_bh_sim.sh
```

## Notes

- Default output is a 1-minute animation at 20 FPS (1200 frames).
- Use `--points-per-bh` to trade quality vs speed.
- Adjust contrast with `--rho-gamma`, `--phi-gamma`, and `--phi-scale`.
- CUDA kernel output lives under `frames_multi/` with `hot/`, `cold/`, and `bh/` subfolders.
- The CUDA path uses `multi_bh_cuda.cu` + `animate_multi_bh_cuda.py` for faster generation.
