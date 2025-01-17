import argparse
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import animation


def load_set(root, shape):
    ny, nx = shape
    def load(pattern):
        paths = sorted(Path(root).glob(pattern))
        if not paths:
            raise SystemExit(f"No frames for {root}/{pattern}")
        frames = [np.fromfile(p, dtype=np.float32).reshape(ny, nx) for p in paths]
        return frames, [p.name for p in paths]
    rho, labels = load("rho_*.bin")
    phi, _ = load("phi_*.bin")
    bx, _ = load("bx_*.bin")
    by, _ = load("by_*.bin")
    return rho, phi, bx, by, labels


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="frames_base")
    ap.add_argument("--refine", default="frames_refine")
    ap.add_argument("--shape", nargs=2, type=int, default=[128, 128])
    ap.add_argument("--out", default="dynpar_compare.mp4")
    ap.add_argument("--fps", type=int, default=12)
    ap.add_argument("--stride", type=int, default=6)
    args = ap.parse_args()

    br, bp, bbx, bby, labels = load_set(args.base, args.shape)
    rr, rp, rbx, rby, _ = load_set(args.refine, args.shape)
    n = min(len(br), len(rr))
    br, bp, bbx, bby = br[:n], bp[:n], bbx[:n], bby[:n]
    rr, rp, rbx, rby = rr[:n], rp[:n], rbx[:n], rby[:n]
    labels = labels[:n]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 5))
    im1 = ax1.imshow(br[0], origin="lower", cmap="viridis")
    im2 = ax2.imshow(rr[0], origin="lower", cmap="viridis")
    ax1.set_title("base")
    ax2.set_title("refined")
    for ax in (ax1, ax2):
        ax.set_xlabel("x")
        ax.set_ylabel("y")

    def quiver_from(ax, bx, by, stride):
        ny, nx = bx.shape
        ys = np.arange(0, ny, stride)
        xs = np.arange(0, nx, stride)
        X, Y = np.meshgrid(xs, ys)
        U = bx[::stride, ::stride]
        V = by[::stride, ::stride]
        return ax.quiver(X, Y, U, V, color="white", scale=30, width=0.003)

    q1 = quiver_from(ax1, bbx[0], bby[0], args.stride)
    q2 = quiver_from(ax2, rbx[0], rby[0], args.stride)
    c1 = ax1.contour(bp[0], levels=8, colors="white", linewidths=0.4)
    c2 = ax2.contour(rp[0], levels=8, colors="white", linewidths=0.4)
    t1 = ax1.text(0.02, 0.98, "", transform=ax1.transAxes, color="white",
                  ha="left", va="top", fontsize=9)
    t2 = ax2.text(0.02, 0.98, "", transform=ax2.transAxes, color="white",
                  ha="left", va="top", fontsize=9)

    def update(i):
        nonlocal c1, c2
        im1.set_data(br[i])
        im2.set_data(rr[i])
        q1.set_UVC(bbx[i][::args.stride, ::args.stride],
                   bby[i][::args.stride, ::args.stride])
        q2.set_UVC(rbx[i][::args.stride, ::args.stride],
                   rby[i][::args.stride, ::args.stride])
        for col in c1.collections:
            col.remove()
        for col in c2.collections:
            col.remove()
        c1 = ax1.contour(bp[i], levels=8, colors="white", linewidths=0.4)
        c2 = ax2.contour(rp[i], levels=8, colors="white", linewidths=0.4)
        bmag1 = np.sqrt(bbx[i]**2 + bby[i]**2).mean()
        bmag2 = np.sqrt(rbx[i]**2 + rby[i]**2).mean()
        t1.set_text(f"|B| mean={bmag1:.3e}")
        t2.set_text(f"|B| mean={bmag2:.3e}")
        fig.suptitle(labels[i])
        return (im1, im2, q1, q2)

    ani = animation.FuncAnimation(fig, update, frames=n, interval=1000/args.fps, blit=False)
    ani.save(args.out, writer="ffmpeg", dpi=120)


if __name__ == "__main__":
    main()
