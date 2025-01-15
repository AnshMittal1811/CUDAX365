import argparse
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import animation, cm, colors
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
import re


def load_frames(pattern, shape):
    paths = sorted(Path(".").glob(pattern))
    if not paths:
        raise SystemExit(f"No frames match pattern: {pattern}")
    ny, nx = shape
    frames = []
    labels = []
    for p in paths:
        raw = np.fromfile(p, dtype=np.float32)
        if raw.size != ny * nx:
            raise SystemExit(
                f"Frame {p} has {raw.size} floats; expected {ny*nx}"
            )
        frames.append(raw.reshape(ny, nx))
        labels.append(p.name)
    return frames, labels


def label_time(label, index, dt):
    match = re.search(r"(\d+)", label)
    if match:
        step = int(match.group(1))
    else:
        step = index
    return step * dt


def main():
    parser = argparse.ArgumentParser(
        description="Animate exact vs rcp.approx MHD outputs."
    )
    parser.add_argument(
        "--exact-rho",
        default="frames_exact/rho_*.bin",
        help="Glob pattern for exact rho frames.",
    )
    parser.add_argument(
        "--exact-phi",
        default="frames_exact/phi_*.bin",
        help="Glob pattern for exact phi frames.",
    )
    parser.add_argument(
        "--approx-rho",
        default="frames_approx/rho_*.bin",
        help="Glob pattern for approx rho frames.",
    )
    parser.add_argument(
        "--approx-phi",
        default="frames_approx/phi_*.bin",
        help="Glob pattern for approx phi frames.",
    )
    parser.add_argument(
        "--shape",
        nargs=2,
        type=int,
        default=[128, 128],
        metavar=("NY", "NX"),
        help="Frame shape for raw .bin files.",
    )
    parser.add_argument(
        "--out", default="mhd_rcp_compare.mp4", help="Output video filename."
    )
    parser.add_argument(
        "--mode",
        choices=["compare", "delta"],
        default="compare",
        help="compare: side-by-side 3D; delta: rho difference heatmap.",
    )
    parser.add_argument(
        "--title", default="Exact vs rcp.approx", help="Figure title."
    )
    parser.add_argument(
        "--fps", type=int, default=12, help="Frames per second."
    )
    parser.add_argument(
        "--dt",
        type=float,
        default=1.0,
        help="Time step per frame for labeling (default 1.0).",
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=2,
        help="Stride for surface plotting (>=1). Higher is faster.",
    )
    parser.add_argument(
        "--elev",
        type=float,
        default=30.0,
        help="Elevation angle for 3D view.",
    )
    parser.add_argument(
        "--azim",
        type=float,
        default=-60.0,
        help="Azimuth angle for 3D view.",
    )
    args = parser.parse_args()

    exact_rho, labels = load_frames(args.exact_rho, args.shape)
    exact_phi, _ = load_frames(args.exact_phi, args.shape)
    approx_rho, _ = load_frames(args.approx_rho, args.shape)
    approx_phi, _ = load_frames(args.approx_phi, args.shape)

    n = min(len(exact_rho), len(approx_rho))
    exact_rho = exact_rho[:n]
    exact_phi = exact_phi[:n]
    approx_rho = approx_rho[:n]
    approx_phi = approx_phi[:n]
    labels = labels[:n]

    rho_min = min(min(f.min() for f in exact_rho), min(f.min() for f in approx_rho))
    rho_max = max(max(f.max() for f in exact_rho), max(f.max() for f in approx_rho))
    phi_min = min(min(f.min() for f in exact_phi), min(f.min() for f in approx_phi))
    phi_max = max(max(f.max() for f in exact_phi), max(f.max() for f in approx_phi))

    if args.mode == "delta":
        fig, ax = plt.subplots(figsize=(6, 5))
        diff0 = approx_rho[0] - exact_rho[0]
        vmax = max(abs(diff0.min()), abs(diff0.max()))
        im = ax.imshow(diff0, origin="lower", cmap="coolwarm",
                       vmin=-vmax, vmax=vmax)
        ax.set_title("rho approx - exact")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        fig.colorbar(im, ax=ax)

        def update(i):
            diff = approx_rho[i] - exact_rho[i]
            vmax = max(abs(diff.min()), abs(diff.max()))
            im.set_data(diff)
            im.set_clim(-vmax, vmax)
            t = label_time(labels[i], i, args.dt)
            fig.suptitle(f"{args.title}  t={t:.3f}")
            return (im,)

        ani = animation.FuncAnimation(
            fig, update, frames=len(exact_rho), interval=1000 / args.fps, blit=True
        )
    else:
        stride = max(1, args.stride)
        ny, nx = exact_rho[0].shape
        x = np.linspace(0, 1, nx)[::stride]
        y = np.linspace(0, 1, ny)[::stride]
        X, Y = np.meshgrid(x, y)

        fig = plt.figure(figsize=(12, 5))
        ax1 = fig.add_subplot(1, 2, 1, projection="3d")
        ax2 = fig.add_subplot(1, 2, 2, projection="3d")
        for ax in (ax1, ax2):
            ax.set_xlabel("x")
            ax.set_ylabel("y")
            ax.set_zlabel("rho")
            ax.set_zlim(rho_min, rho_max)
            ax.view_init(elev=args.elev, azim=args.azim)

        norm_phi = colors.Normalize(vmin=phi_min, vmax=phi_max)
        cmap_phi = cm.magma
        sm = cm.ScalarMappable(norm=norm_phi, cmap=cmap_phi)
        sm.set_array([])
        fig.colorbar(sm, ax=[ax1, ax2], shrink=0.6, pad=0.05, label="phi")

        surf1 = [None]
        surf2 = [None]

        def update(i):
            if surf1[0] is not None:
                surf1[0].remove()
            if surf2[0] is not None:
                surf2[0].remove()
            rho_e = exact_rho[i][::stride, ::stride]
            phi_e = exact_phi[i][::stride, ::stride]
            rho_a = approx_rho[i][::stride, ::stride]
            phi_a = approx_phi[i][::stride, ::stride]
            surf1[0] = ax1.plot_surface(
                X, Y, rho_e,
                facecolors=cmap_phi(norm_phi(phi_e)),
                rstride=1, cstride=1, linewidth=0, antialiased=False, shade=False
            )
            surf2[0] = ax2.plot_surface(
                X, Y, rho_a,
                facecolors=cmap_phi(norm_phi(phi_a)),
                rstride=1, cstride=1, linewidth=0, antialiased=False, shade=False
            )
            t = label_time(labels[i], i, args.dt)
            fig.suptitle(f"{args.title}  t={t:.3f}")
            ax1.set_title("exact (1/x)")
            ax2.set_title("rcp.approx")
            return []

        ani = animation.FuncAnimation(
            fig, update, frames=len(exact_rho), interval=1000 / args.fps, blit=False
        )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    writer = "pillow" if out_path.suffix.lower() == ".gif" else "ffmpeg"
    ani.save(out_path, writer=writer, dpi=120)
    plt.close(fig)

    print(f"Saved animation to {out_path} using {len(exact_rho)} frames.")


if __name__ == "__main__":
    main()
