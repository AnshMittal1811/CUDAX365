
import argparse
import json
import math
from pathlib import Path

import numpy as np


def make_grid(nx, ny):
    x = np.linspace(-1.0, 1.0, nx, dtype=np.float32)
    y = np.linspace(-1.0, 1.0, ny, dtype=np.float32)
    xx, yy = np.meshgrid(x, y)
    r = np.sqrt(xx * xx + yy * yy)
    theta = np.arctan2(yy, xx)
    return xx, yy, r, theta


def compute_fields(r, theta, t, params):
    gm = params['gm']
    rs = params['rs']
    r_safe = np.maximum(r, rs + 1e-3)
    phi_grav = -gm / (r_safe - rs)
    omega = np.sqrt(gm / (r_safe * r_safe * r_safe))
    phase = theta - omega * t
    rho = (
        params['rho0']
        * np.exp(-((r - params['r0']) ** 2) / (2.0 * params['sigma'] ** 2))
        * (1.0 + params['arm_amp'] * np.sin(params['m'] * phase))
    )
    phi = phi_grav + params['phi_amp'] * np.cos(params['m'] * phase)
    return rho.astype(np.float32), phi.astype(np.float32)


def main():
    ap = argparse.ArgumentParser(description='Generate black hole flow frames')
    ap.add_argument('--nx', type=int, default=160)
    ap.add_argument('--ny', type=int, default=160)
    ap.add_argument('--frames', type=int, default=1440)
    ap.add_argument('--dt', type=float, default=0.05)
    ap.add_argument('--out-hot', type=str, default='frames_hot')
    ap.add_argument('--out-cold', type=str, default='frames_cold')
    args = ap.parse_args()

    out_hot = Path(args.out_hot)
    out_cold = Path(args.out_cold)
    out_hot.mkdir(parents=True, exist_ok=True)
    out_cold.mkdir(parents=True, exist_ok=True)

    xx, yy, r, theta = make_grid(args.nx, args.ny)

    hot = {
        'gm': 1.0,
        'rs': 0.18,
        'rho0': 1.2,
        'r0': 0.45,
        'sigma': 0.12,
        'arm_amp': 0.35,
        'm': 3,
        'phi_amp': 0.8,
    }
    cold = {
        'gm': 1.0,
        'rs': 0.18,
        'rho0': 0.9,
        'r0': 0.62,
        'sigma': 0.16,
        'arm_amp': 0.25,
        'm': 2,
        'phi_amp': 0.6,
    }

    for i in range(args.frames):
        t = i * args.dt
        rho_h, phi_h = compute_fields(r, theta, t, hot)
        rho_c, phi_c = compute_fields(r, theta, t, cold)

        rho_h.tofile(out_hot / f'rho_{i:04d}.bin')
        phi_h.tofile(out_hot / f'phi_{i:04d}.bin')
        rho_c.tofile(out_cold / f'rho_{i:04d}.bin')
        phi_c.tofile(out_cold / f'phi_{i:04d}.bin')

        if i % 100 == 0:
            print(f'Wrote frame {i}/{args.frames}')

    meta = {
        'nx': args.nx,
        'ny': args.ny,
        'frames': args.frames,
        'dt': args.dt,
        'hot_params': hot,
        'cold_params': cold,
    }
    with open('frame_meta.json', 'w', encoding='utf-8') as f:
        json.dump(meta, f, indent=2)
    print('Saved frame_meta.json')


if __name__ == '__main__':
    main()
