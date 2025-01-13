import argparse
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import animation
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401


def select_field(arr, field):
    if field == "mag":
        return np.abs(arr)
    if field == "phase":
        return np.angle(arr)
    if field == "real":
        return arr.real
    if field == "imag":
        return arr.imag
    return np.abs(arr)


def load_frames(pattern, shape, field):
    paths = sorted(Path(".").glob(pattern))
    if not paths:
        raise SystemExit(f"No frames match pattern: {pattern}")
    ny, nx = shape
    frames = []
    labels = []
    for p in paths:
        if p.suffix.lower() == ".npy":
            arr = np.load(p)
            if np.iscomplexobj(arr):
                arr = select_field(arr, field)
            if arr.ndim != 2:
                raise SystemExit(f"Frame {p} is not 2D (shape {arr.shape})")
        else:
            raw = np.fromfile(p, dtype=np.float32)
            if raw.size == ny * nx * 2:
                arr = raw.reshape(ny, nx, 2)
                comp = arr[..., 0] + 1j * arr[..., 1]
                arr = select_field(comp, field)
            elif raw.size == ny * nx:
                arr = raw.reshape(ny, nx)
            else:
                raise SystemExit(
                    f"Frame {p} has {raw.size} floats; expected {ny*nx} or {ny*nx*2}"
                )
        frames.append(arr.astype(np.float32))
        labels.append(p.name)
    return frames, labels


def compute_limits(frames, field):
    if field == "phase":
        return -np.pi, np.pi
    vmin = min(f.min() for f in frames)
    vmax = max(f.max() for f in frames)
    return vmin, vmax


def make_heatmap(frames, labels, out_path, title, fps, vmin, vmax):
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(frames[0], origin="lower", cmap="viridis", vmin=vmin, vmax=vmax)
    ax.set_title(title)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.grid(True, color="white", alpha=0.2)
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("value")

    def update(i):
        im.set_data(frames[i])
        ax.set_title(f"{title}  ({labels[i]})")
        return (im,)

    ani = animation.FuncAnimation(
        fig, update, frames=len(frames), interval=1000 / fps, blit=True
    )
    writer = "pillow" if out_path.suffix.lower() == ".gif" else "ffmpeg"
    ani.save(out_path, writer=writer, dpi=120)
    plt.close(fig)


def make_surface(frames, labels, out_path, title, fps, vmin, vmax):
    ny, nx = frames[0].shape
    x = np.linspace(0, 1, nx)
    y = np.linspace(0, 1, ny)
    X, Y = np.meshgrid(x, y)

    fig = plt.figure(figsize=(7, 6))
    ax = fig.add_subplot(111, projection="3d")
    surf = [ax.plot_surface(X, Y, frames[0], cmap="viridis", vmin=vmin, vmax=vmax)]
    ax.set_zlim(vmin, vmax)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("value")
    ax.set_title(title)

    def update(i):
        ax.collections.clear()
        surf[0] = ax.plot_surface(
            X, Y, frames[i], cmap="viridis", vmin=vmin, vmax=vmax
        )
        ax.set_title(f"{title}  ({labels[i]})")
        return surf

    ani = animation.FuncAnimation(
        fig, update, frames=len(frames), interval=1000 / fps, blit=False
    )
    writer = "pillow" if out_path.suffix.lower() == ".gif" else "ffmpeg"
    ani.save(out_path, writer=writer, dpi=120)
    plt.close(fig)


def interpolate_frames(frames, labels, n_interp):
    if n_interp <= 1 or len(frames) < 2:
        return frames, labels
    out_frames = []
    out_labels = []
    for i in range(len(frames) - 1):
        f0, f1 = frames[i], frames[i + 1]
        out_frames.append(f0)
        out_labels.append(labels[i])
        for k in range(1, n_interp):
            alpha = k / float(n_interp)
            out_frames.append((1 - alpha) * f0 + alpha * f1)
            out_labels.append(f"{labels[i]}->{labels[i+1]}[{alpha:.2f}]")
    out_frames.append(frames[-1])
    out_labels.append(labels[-1])
    return out_frames, out_labels


def main():
    parser = argparse.ArgumentParser(
        description="Animate FFT stage frames dumped by butterfly_fft.cu."
    )
    parser.add_argument(
        "--pattern",
        default="frames/fft_stage_*.bin",
        help="Glob pattern for frame files.",
    )
    parser.add_argument(
        "--mode",
        choices=["heatmap", "surface"],
        default="heatmap",
        help="Render as 2D heatmap or 3D surface.",
    )
    parser.add_argument(
        "--field",
        choices=["mag", "phase", "real", "imag"],
        default="mag",
        help="Field to visualize when frames are complex.",
    )
    parser.add_argument(
        "--shape",
        nargs=2,
        type=int,
        default=[32, 32],
        metavar=("NY", "NX"),
        help="Frame shape for raw .bin files.",
    )
    parser.add_argument(
        "--out", default="fft_anim.mp4", help="Output video filename."
    )
    parser.add_argument(
        "--title", default="FFT stages", help="Title/heading for the plot."
    )
    parser.add_argument(
        "--fps", type=int, default=10, help="Frames per second for the animation."
    )
    parser.add_argument(
        "--interp",
        type=int,
        default=1,
        help="Number of interpolated steps between frames (>=1). Higher = smoother.",
    )
    args = parser.parse_args()

    frames, labels = load_frames(args.pattern, args.shape, args.field)
    frames, labels = interpolate_frames(frames, labels, args.interp)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    vmin, vmax = compute_limits(frames, args.field)
    if args.mode == "heatmap":
        make_heatmap(frames, labels, out_path, args.title, args.fps, vmin, vmax)
    else:
        make_surface(frames, labels, out_path, args.title, args.fps, vmin, vmax)

    print(f"Saved animation to {out_path} using {len(frames)} frames.")


if __name__ == "__main__":
    main()
