import argparse
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import animation
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401


def load_frames(pattern):
    paths = sorted(Path(".").glob(pattern))
    if not paths:
        raise SystemExit(f"No frames match pattern: {pattern}")
    frames = []
    for p in paths:
        arr = np.load(p)
        if arr.ndim != 2:
            raise SystemExit(f"Frame {p} is not 2D (shape {arr.shape})")
        frames.append(arr)
    return frames, paths


def make_heatmap(frames, labels, out_path, title, fps):
    vmin = min(f.min() for f in frames)
    vmax = max(f.max() for f in frames)
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


def make_surface(frames, labels, out_path, title, fps):
    vmin = min(f.min() for f in frames)
    vmax = max(f.max() for f in frames)
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


def interpolate_frames(frames, paths, n_interp):
    if n_interp <= 1 or len(frames) < 2:
        return frames, [p.name for p in paths]
    out_frames = []
    out_labels = []
    for i in range(len(frames) - 1):
        f0, f1 = frames[i], frames[i + 1]
        out_frames.append(f0)
        out_labels.append(paths[i].name)
        for k in range(1, n_interp):
            alpha = k / float(n_interp)
            out_frames.append((1 - alpha) * f0 + alpha * f1)
            out_labels.append(f"{paths[i].name}→{paths[i+1].name}[{alpha:.2f}]")
    out_frames.append(frames[-1])
    out_labels.append(paths[-1].name)
    return out_frames, out_labels


def main():
    parser = argparse.ArgumentParser(
        description="Animate 2D fields (rho/phi/etc) dumped as .npy frames."
    )
    parser.add_argument(
        "--pattern",
        default="frames/rho_*.npy",
        help="Glob pattern for frame files (2D float arrays).",
    )
    parser.add_argument(
        "--mode",
        choices=["heatmap", "surface"],
        default="heatmap",
        help="Render as 2D heatmap or 3D surface.",
    )
    parser.add_argument(
        "--out", default="poisson_anim.mp4", help="Output video filename."
    )
    parser.add_argument(
        "--title", default="Poisson field", help="Title/heading for the plot."
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

    frames, paths = load_frames(args.pattern)
    frames, labels = interpolate_frames(frames, paths, args.interp)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if args.mode == "heatmap":
        make_heatmap(frames, labels, out_path, args.title, args.fps)
    else:
        make_surface(frames, labels, out_path, args.title, args.fps)

    print(f"Saved animation to {out_path} using {len(frames)} frames.")


if __name__ == "__main__":
    main()
