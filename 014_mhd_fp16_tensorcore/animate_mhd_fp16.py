import argparse
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import animation


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


def main():
    parser = argparse.ArgumentParser(
        description="Animate rho/phi frames side-by-side."
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
        "--out", default="mhd_rho_phi.mp4", help="Output video filename."
    )
    parser.add_argument(
        "--title", default="FP16 MHD (rho | phi)", help="Figure title."
    )
    parser.add_argument(
        "--rho-title", default="rho", help="Title for the rho panel."
    )
    parser.add_argument(
        "--phi-title", default="phi", help="Title for the phi panel."
    )
    parser.add_argument(
        "--fps", type=int, default=10, help="Frames per second."
    )
    parser.add_argument(
        "--interp",
        type=int,
        default=1,
        help="Interpolated steps between frames (>=1).",
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
        ax1.set_title(f"{args.rho_title}  ({labels[i]})")
        ax2.set_title(f"{args.phi_title}  ({labels[i]})")
        return (im1, im2)

    ani = animation.FuncAnimation(
        fig, update, frames=len(rho_frames), interval=1000 / args.fps, blit=True
    )
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    writer = "pillow" if out_path.suffix.lower() == ".gif" else "ffmpeg"
    ani.save(out_path, writer=writer, dpi=120)
    plt.close(fig)

    print(f"Saved animation to {out_path} using {len(rho_frames)} frames.")


if __name__ == "__main__":
    main()
