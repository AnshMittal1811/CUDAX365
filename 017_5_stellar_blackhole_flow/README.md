
# Day 17.5: Stellar Material Flow Around a Black Hole

This folder generates a synthetic accretion flow around a stellar-mass black hole and renders
a side-by-side 3D visualization for two distinct fluids (rho/phi pairs).

Governing equations (parametric model used for visualization):

- Pseudo-Newtonian potential (Paczynski-Wiita):
  phi_grav(r) = -G*M / (r - r_s), where r_s = 2*G*M/c^2

- Angular velocity (Kepler-like):
  Omega(r) = sqrt(G*M / (r^3))

- Density field (fluid i):
  rho_i(r, theta, t) = rho0_i * exp(-(r - r0_i)^2 / (2*sigma_i^2))
                       * (1 + a_i * sin(m_i * (theta - Omega(r)*t)))

- Scalar phi field (fluid i):
  phi_i(r, theta, t) = phi_grav(r) + b_i * cos(m_i * (theta - Omega(r)*t))

The fields are synthetic but capture a swirling, shearing disk structure around an event horizon.

Run (from this folder):
```bash
./run_blackhole_sim.sh
```

Notes:
- Default output is a 2-minute animation at 12 FPS (1440 frames).
- Default resolution is 160x160 with stride 2 in the renderer to keep size/time reasonable.
- The animation colors the rho surface using phi (symlog scale by default).
- If `nvcc` is available, the CUDA generator is used; set `ARCH=sm_89` or similar for your GPU.
- If you want a faster preview, reduce --frames or --fps in run_blackhole_sim.sh.
