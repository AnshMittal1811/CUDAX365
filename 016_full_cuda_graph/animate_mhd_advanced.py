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


def interpolate_frames(frames_a, frames_b, labels, n_interp):
    if n_interp <= 1 or len(frames_a) < 2:
        return frames_a, frames_b, labels
    out_a, out_b, out_labels = [], [], []
    for i in range(len(frames_a) - 1):
        a0, a1 = frames_a[i], frames_a[i + 1]
        b0, b1 = frames_b[i], frames_b[i + 1]
        out_a.append(a0)
        out_b.append(b0)
        out_labels.append(labels[i])
        for k in range(1, n_interp):
            alpha = k / float(n_interp)
            out_a.append((1 - alpha) * a0 + alpha * a1)
            out_b.append((1 - alpha) * b0 + alpha * b1)
            out_labels.append(f"{labels[i]}->{labels[i+1]}[{alpha:.2f}]")
    out_a.append(frames_a[-1])
    out_b.append(frames_b[-1])
    out_labels.append(labels[-1])
    return out_a, out_b, out_labels


def label_time(label, index, dt):
    match = re.search(r"(\d+)", label)
    if match:
        step = int(match.group(1))
    else:
        step = index
    return step * dt


def main():
    parser = argparse.ArgumentParser(
        description="Animate rho/phi with combined 3D visualization."
    )
    parser.add_argument(
        "--rho",
        default="frames/rho_*.bin",
        help="Glob pattern for rho frames.",
    )
    parser.add_argument(
        "--phi",
        default="frames/phi_*.bin",
        help="Glob pattern for phi frames.",
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
        "--out", default="mhd_combined_3d.mp4", help="Output video filename."
    )
    parser.add_argument(
        "--mode",
        choices=["combined", "side-by-side"],
        default="combined",
        help="combined: rho surface colored by phi; side-by-side: heatmaps.",
    )
    parser.add_argument(
        "--title", default="Advanced MHD (rho colored by phi)", help="Figure title."
    )
    parser.add_argument(
        "--rho-title", default="rho", help="Title for rho field."
    )
    parser.add_argument(
        "--phi-title", default="phi", help="Title for phi field."
    )
    parser.add_argument(
        "--fps", type=int, default=12, help="Frames per second."
    )
    parser.add_argument(
        "--interp",
        type=int,
        default=1,
        help="Interpolated steps between frames (>=1).",
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

    rho_frames, rho_labels = load_frames(args.rho, args.shape)
    phi_frames, phi_labels = load_frames(args.phi, args.shape)
    if len(rho_frames) != len(phi_frames):
        raise SystemExit("rho/phi frame counts do not match.")
    labels = rho_labels
    rho_frames, phi_frames, labels = interpolate_frames(
        rho_frames, phi_frames, labels, args.interp
    )

    rho_min = min(f.min() for f in rho_frames)
    rho_max = max(f.max() for f in rho_frames)
    phi_min = min(f.min() for f in phi_frames)
    phi_max = max(f.max() for f in phi_frames)

    if args.mode == "side-by-side":
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
        fig.suptitle(args.title)
        im1 = ax1.imshow(rho_frames[0], origin="lower", cmap="viridis",
                         vmin=rho_min, vmax=rho_max)
        im2 = ax2.imshow(phi_frames[0], origin="lower", cmap="magma",
                         vmin=phi_min, vmax=phi_max)
        ax1.set_title(args.rho_title)
        ax2.set_title(args.phi_title)
        ax1.set_xlabel("x")
        ax1.set_ylabel("y")
        ax2.set_xlabel("x")
        ax2.set_ylabel("y")
        fig.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)
        fig.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)

        def update(i):
            im1.set_data(rho_frames[i])
            im2.set_data(phi_frames[i])
            t = label_time(labels[i], i, args.dt)
            fig.suptitle(f"{args.title}  t={t:.3f}")
            ax1.set_title(f"{args.rho_title}  ({labels[i]})")
            ax2.set_title(f"{args.phi_title}  ({labels[i]})")
            return (im1, im2)

        ani = animation.FuncAnimation(
            fig, update, frames=len(rho_frames), interval=1000 / args.fps, blit=True
        )
    else:
        stride = max(1, args.stride)
        ny, nx = rho_frames[0].shape
        x = np.linspace(0, 1, nx)[::stride]
        y = np.linspace(0, 1, ny)[::stride]
        X, Y = np.meshgrid(x, y)

        fig = plt.figure(figsize=(7, 5))
        ax = fig.add_subplot(111, projection="3d")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel(args.rho_title)
        ax.set_zlim(rho_min, rho_max)
        ax.view_init(elev=args.elev, azim=args.azim)
        ax.set_title(args.rho_title)

        norm_phi = colors.Normalize(vmin=phi_min, vmax=phi_max)
        cmap_phi = cm.magma
        sm = cm.ScalarMappable(norm=norm_phi, cmap=cmap_phi)
        sm.set_array([])
        fig.colorbar(sm, ax=ax, shrink=0.6, pad=0.08, label=args.phi_title)

        surf = [None]

        def update(i):
            if surf[0] is not None:
                surf[0].remove()
            rho = rho_frames[i][::stride, ::stride]
            phi = phi_frames[i][::stride, ::stride]
            colors_phi = cmap_phi(norm_phi(phi))
            surf[0] = ax.plot_surface(
                X, Y, rho,
                facecolors=colors_phi,
                rstride=1, cstride=1,
                linewidth=0, antialiased=False, shade=False
            )
            t = label_time(labels[i], i, args.dt)
            fig.suptitle(f"{args.title}  t={t:.3f}")
            ax.set_title(f"{args.rho_title}  ({labels[i]})")
            return []

        ani = animation.FuncAnimation(
            fig, update, frames=len(rho_frames), interval=1000 / args.fps, blit=False
        )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    writer = "pillow" if out_path.suffix.lower() == ".gif" else "ffmpeg"
    ani.save(out_path, writer=writer, dpi=120)
    plt.close(fig)

    print(f"Saved animation to {out_path} using {len(rho_frames)} frames.")


if __name__ == "__main__":
    main()
