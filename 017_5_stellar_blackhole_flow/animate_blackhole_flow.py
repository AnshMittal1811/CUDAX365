
import argparse
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import animation, cm, colors
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401


def load_paths(pattern):
    paths = sorted(Path('.').glob(pattern))
    if not paths:
        raise SystemExit(f'No frames for pattern: {pattern}')
    return paths


def load_frame(path, shape):
    raw = np.fromfile(path, dtype=np.float32)
    if raw.size != shape[0] * shape[1]:
        raise SystemExit(f'{path} has {raw.size} floats; expected {shape[0] * shape[1]}')
    return raw.reshape(shape)


def scan_range(paths, shape, step):
    vmin = float('inf')
    vmax = float('-inf')
    for path in paths[::step]:
        arr = load_frame(path, shape)
        vmin = min(vmin, float(arr.min()))
        vmax = max(vmax, float(arr.max()))
    if not np.isfinite(vmin) or not np.isfinite(vmax):
        vmin, vmax = 0.0, 1.0
    return vmin, vmax


def draw_black_hole(ax, radius=0.15):
    u = np.linspace(0.0, 2.0 * np.pi, 40)
    v = np.linspace(0.0, np.pi, 20)
    x = radius * np.outer(np.cos(u), np.sin(v))
    y = radius * np.outer(np.sin(u), np.sin(v))
    z = radius * np.outer(np.ones_like(u), np.cos(v))
    ax.plot_surface(x, y, z, color='black', alpha=0.9, linewidth=0.0)


def main():
    ap = argparse.ArgumentParser(description='Render black hole flow side-by-side')
    ap.add_argument('--hot-rho', default='frames_hot/rho_*.bin')
    ap.add_argument('--hot-phi', default='frames_hot/phi_*.bin')
    ap.add_argument('--cold-rho', default='frames_cold/rho_*.bin')
    ap.add_argument('--cold-phi', default='frames_cold/phi_*.bin')
    ap.add_argument('--shape', nargs=2, type=int, default=[160, 160], metavar=('NY', 'NX'))
    ap.add_argument('--stride', type=int, default=2)
    ap.add_argument('--fps', type=int, default=12)
    ap.add_argument('--out', default='blackhole_flow_3d.mp4')
    ap.add_argument('--scan-step', type=int, default=12)
    ap.add_argument('--zscale', type=float, default=0.6)
    ap.add_argument('--phi-scale', choices=['linear', 'symlog'], default='symlog')
    ap.add_argument('--phi-linthresh', type=float, default=0.0)
    ap.add_argument('--title', default='Stellar Black Hole Accretion Flow (rho/phi)')
    args = ap.parse_args()

    shape = tuple(args.shape)
    hot_rho_paths = load_paths(args.hot_rho)
    hot_phi_paths = load_paths(args.hot_phi)
    cold_rho_paths = load_paths(args.cold_rho)
    cold_phi_paths = load_paths(args.cold_phi)

    frame_count = min(len(hot_rho_paths), len(cold_rho_paths), len(hot_phi_paths), len(cold_phi_paths))
    hot_rho_paths = hot_rho_paths[:frame_count]
    hot_phi_paths = hot_phi_paths[:frame_count]
    cold_rho_paths = cold_rho_paths[:frame_count]
    cold_phi_paths = cold_phi_paths[:frame_count]

    rho_min_h, rho_max_h = scan_range(hot_rho_paths, shape, args.scan_step)
    rho_min_c, rho_max_c = scan_range(cold_rho_paths, shape, args.scan_step)
    phi_min_h, phi_max_h = scan_range(hot_phi_paths, shape, args.scan_step)
    phi_min_c, phi_max_c = scan_range(cold_phi_paths, shape, args.scan_step)

    rho_min = min(rho_min_h, rho_min_c)
    rho_max = max(rho_max_h, rho_max_c)
    phi_min = min(phi_min_h, phi_min_c)
    phi_max = max(phi_max_h, phi_max_c)

    cmap = cm.magma
    if args.phi_scale == 'symlog':
        linthresh = args.phi_linthresh
        if linthresh <= 0.0:
            linthresh = max(1e-3, (phi_max - phi_min) * 0.02)
        norm = colors.SymLogNorm(
            linthresh=linthresh,
            linscale=1.0,
            vmin=phi_min,
            vmax=phi_max,
            base=10,
        )
        phi_label = 'phi (symlog)'
    else:
        norm = colors.Normalize(vmin=phi_min, vmax=phi_max)
        phi_label = 'phi'

    ny, nx = shape
    stride = max(1, args.stride)
    x = np.linspace(-1.0, 1.0, nx)[::stride]
    y = np.linspace(-1.0, 1.0, ny)[::stride]
    X, Y = np.meshgrid(x, y)

    fig = plt.figure(figsize=(12, 5))
    ax1 = fig.add_subplot(1, 2, 1, projection='3d')
    ax2 = fig.add_subplot(1, 2, 2, projection='3d')
    ax1.set_title('hot flow')
    ax2.set_title('cold flow')

    for ax in (ax1, ax2):
        ax.set_xlim(-1.0, 1.0)
        ax.set_ylim(-1.0, 1.0)
        ax.set_zlim(0.0, (rho_max - rho_min) * args.zscale + 0.2)
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('rho')
        ax.view_init(elev=28.0, azim=-60.0)
        ax.set_box_aspect((1.0, 1.0, 0.6))
        draw_black_hole(ax)

    sm = cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    fig.colorbar(sm, ax=[ax1, ax2], shrink=0.6, pad=0.08, label=phi_label)
    fig.suptitle(args.title)

    surf1 = [None]
    surf2 = [None]

    def update(i):
        rho_h = load_frame(hot_rho_paths[i], shape)[::stride, ::stride]
        phi_h = load_frame(hot_phi_paths[i], shape)[::stride, ::stride]
        rho_c = load_frame(cold_rho_paths[i], shape)[::stride, ::stride]
        phi_c = load_frame(cold_phi_paths[i], shape)[::stride, ::stride]

        z_h = (rho_h - rho_min) * args.zscale
        z_c = (rho_c - rho_min) * args.zscale

        if surf1[0] is not None:
            surf1[0].remove()
        if surf2[0] is not None:
            surf2[0].remove()

        surf1[0] = ax1.plot_surface(
            X, Y, z_h,
            facecolors=cmap(norm(phi_h)),
            rstride=1, cstride=1,
            linewidth=0.0, antialiased=False, shade=False
        )
        surf2[0] = ax2.plot_surface(
            X, Y, z_c,
            facecolors=cmap(norm(phi_c)),
            rstride=1, cstride=1,
            linewidth=0.0, antialiased=False, shade=False
        )
        return surf1[0], surf2[0]

    ani = animation.FuncAnimation(fig, update, frames=frame_count, interval=1000 / args.fps)
    writer = animation.FFMpegWriter(fps=args.fps, bitrate=1800)
    ani.save(args.out, writer=writer, dpi=120)
    print(f'Saved {args.out}')


if __name__ == '__main__':
    main()
