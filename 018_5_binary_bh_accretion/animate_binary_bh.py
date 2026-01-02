import argparse
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import animation
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401


def load_paths(pattern):
    paths = sorted(Path('.').glob(pattern))
    if not paths:
        raise SystemExit(f"No frames for pattern: {pattern}")
    return paths


def load_frame(path, shape):
    raw = np.fromfile(path, dtype=np.float32)
    if raw.size != shape[0] * shape[1]:
        raise SystemExit(f"{path} has {raw.size} floats; expected {shape[0] * shape[1]}")
    return raw.reshape(shape)


def normalize_field(field, scale_mode):
    if scale_mode == "symlog":
        field = np.sign(field) * np.log1p(np.abs(field))
    fmin = float(field.min())
    fmax = float(field.max())
    if fmax - fmin < 1e-6:
        return np.zeros_like(field)
    return (field - fmin) / (fmax - fmin)


def mix_colors(rho_norm, phi_norm, rho_color, phi_color):
    mix = rho_norm[..., None] * rho_color + phi_norm[..., None] * phi_color
    return np.clip(mix, 0.0, 1.0)


def main():
    ap = argparse.ArgumentParser(description="Render binary black hole accretion flow")
    ap.add_argument("--rho", default="frames_binary/rho_*.bin")
    ap.add_argument("--phi", default="frames_binary/phi_*.bin")
    ap.add_argument("--shape", nargs=2, type=int, default=[192, 192], metavar=("NY", "NX"))
    ap.add_argument("--stride", type=int, default=2)
    ap.add_argument("--fps", type=int, default=12)
    ap.add_argument("--out", default="binary_bh_3d.mp4")
    ap.add_argument("--zscale", type=float, default=0.6)
    ap.add_argument("--phi-scale", choices=["linear", "symlog"], default="symlog")
    ap.add_argument("--rho-gamma", type=float, default=1.0)
    ap.add_argument("--phi-gamma", type=float, default=1.0)
    ap.add_argument("--title", default="Binary Black Hole Accretion (rho+phi mix)")
    args = ap.parse_args()

    shape = tuple(args.shape)
    rho_paths = load_paths(args.rho)
    phi_paths = load_paths(args.phi)
    frame_count = min(len(rho_paths), len(phi_paths))
    rho_paths = rho_paths[:frame_count]
    phi_paths = phi_paths[:frame_count]

    stride = max(1, args.stride)
    ny, nx = shape
    x = np.linspace(-1.0, 1.0, nx)[::stride]
    y = np.linspace(-1.0, 1.0, ny)[::stride]
    X, Y = np.meshgrid(x, y)

    rho_color = np.array([0.15, 0.55, 1.0])
    phi_color = np.array([1.0, 0.45, 0.1])

    fig = plt.figure(figsize=(7.5, 6))
    ax = fig.add_subplot(1, 1, 1, projection="3d")
    ax.set_xlim(-1.0, 1.0)
    ax.set_ylim(-1.0, 1.0)
    ax.set_zlim(0.0, args.zscale * 1.5)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("rho")
    ax.view_init(elev=28.0, azim=-60.0)
    ax.set_box_aspect((1.0, 1.0, 0.6))
    fig.suptitle(args.title)

    surf = [None]

    def update(i):
        rho = load_frame(rho_paths[i], shape)[::stride, ::stride]
        phi = load_frame(phi_paths[i], shape)[::stride, ::stride]

        rho_norm = normalize_field(rho, "linear") ** args.rho_gamma
        phi_norm = normalize_field(phi, args.phi_scale) ** args.phi_gamma
        colors = mix_colors(rho_norm, phi_norm, rho_color, phi_color)

        z = rho_norm * args.zscale

        if surf[0] is not None:
            surf[0].remove()
        surf[0] = ax.plot_surface(
            X, Y, z,
            facecolors=colors,
            rstride=1, cstride=1,
            linewidth=0.0, antialiased=False, shade=False
        )
        return surf[0]

    ani = animation.FuncAnimation(fig, update, frames=frame_count, interval=1000 / args.fps)
    writer = animation.FFMpegWriter(fps=args.fps, bitrate=2000)
    ani.save(args.out, writer=writer, dpi=120)
    print(f"Saved {args.out}")


if __name__ == "__main__":
    main()
