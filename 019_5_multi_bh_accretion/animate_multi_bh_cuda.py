import argparse
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import animation
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401


def load_paths(pattern):
    paths = sorted(Path(".").glob(pattern))
    if not paths:
        raise SystemExit(f"No frames for pattern: {pattern}")
    return paths


def load_frame(path, stride):
    raw = np.fromfile(path, dtype=np.float32)
    if raw.size % stride != 0:
        raise SystemExit(f"{path} has {raw.size} floats; expected stride {stride}")
    return raw.reshape(-1, stride)


def normalize_field(field, mode):
    if mode == "symlog":
        field = np.sign(field) * np.log1p(np.abs(field))
    vmin = float(field.min())
    vmax = float(field.max())
    if vmax - vmin < 1e-6:
        return np.zeros_like(field)
    return (field - vmin) / (vmax - vmin)


def mix_colors(rho_norm, phi_norm, rho_color, phi_color, intensity):
    mix = rho_norm[:, None] * rho_color + phi_norm[:, None] * phi_color
    mix = mix * intensity[:, None]
    return np.clip(mix, 0.0, 1.0)


def make_circle(radius, n=80):
    t = np.linspace(0.0, 2.0 * np.pi, n)
    x = radius * np.cos(t)
    y = radius * np.sin(t)
    z = np.zeros_like(t)
    return x, y, z


def update_line(line, x, y, z):
    line.set_data(x, y)
    line.set_3d_properties(z)


def main():
    ap = argparse.ArgumentParser(description="Animate multi-BH accretion from CUDA frames")
    ap.add_argument("--hot", default="frames_multi/hot/hot_*.bin")
    ap.add_argument("--cold", default="frames_multi/cold/cold_*.bin")
    ap.add_argument("--bh", default="frames_multi/bh/bh_*.bin")
    ap.add_argument("--fps", type=int, default=20)
    ap.add_argument("--out", default="multi_bh_accretion.mp4")
    ap.add_argument("--phi-scale", choices=["linear", "symlog"], default="symlog")
    ap.add_argument("--rho-gamma", type=float, default=1.0)
    ap.add_argument("--phi-gamma", type=float, default=1.0)
    ap.add_argument("--title", default="8-BH Cluster + SMBH Accretion")
    ap.add_argument("--no-bh-lines", action="store_true")
    args = ap.parse_args()

    hot_paths = load_paths(args.hot)
    cold_paths = load_paths(args.cold)
    bh_paths = load_paths(args.bh)

    frame_count = min(len(hot_paths), len(cold_paths), len(bh_paths))
    hot_paths = hot_paths[:frame_count]
    cold_paths = cold_paths[:frame_count]
    bh_paths = bh_paths[:frame_count]

    rho_color = np.array([0.2, 0.6, 1.0])
    phi_color = np.array([1.0, 0.45, 0.1])

    fig = plt.figure(figsize=(12, 8))
    gs = fig.add_gridspec(2, 2, height_ratios=[2.0, 1.6])
    ax_top = fig.add_subplot(gs[0, :], projection="3d")
    ax_hot = fig.add_subplot(gs[1, 0], projection="3d")
    ax_cold = fig.add_subplot(gs[1, 1], projection="3d")

    for ax in (ax_top, ax_hot, ax_cold):
        ax.set_xlim(-1.8, 1.8)
        ax.set_ylim(-1.8, 1.8)
        ax.set_zlim(-0.6, 0.6)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")
        ax.view_init(elev=25.0, azim=-60.0)
        ax.set_box_aspect((1.0, 1.0, 0.45))

    ax_top.set_title("Combined hot+cold flow")
    ax_hot.set_title("Hot flow")
    ax_cold.set_title("Cold flow")
    fig.suptitle(args.title)

    scatter_top = ax_top.scatter([], [], [], s=4, c=[], depthshade=False)
    scatter_hot = ax_hot.scatter([], [], [], s=4, c=[], depthshade=False)
    scatter_cold = ax_cold.scatter([], [], [], s=4, c=[], depthshade=False)

    bh_core_top = ax_top.scatter([], [], [], s=18, c="white", depthshade=False)
    bh_halo_top = ax_top.scatter([], [], [], s=80, c="#66ccff", alpha=0.25, depthshade=False)
    bh_core_hot = ax_hot.scatter([], [], [], s=18, c="white", depthshade=False)
    bh_core_cold = ax_cold.scatter([], [], [], s=18, c="white", depthshade=False)

    smbh_loop = make_circle(0.6)
    smbh_lines = [ax_top.plot([], [], [], color="#55ddff", lw=1.0, alpha=0.6)[0],
                  ax_hot.plot([], [], [], color="#55ddff", lw=1.0, alpha=0.5)[0],
                  ax_cold.plot([], [], [], color="#55ddff", lw=1.0, alpha=0.5)[0]]

    bh_lines = []
    if not args.no_bh_lines:
        for _ in range(8):
            line = ax_top.plot([], [], [], color="#77ffaa", lw=0.8, alpha=0.6)[0]
            bh_lines.append(line)
        bh_loop = make_circle(0.08)

    def update(frame):
        hot = load_frame(hot_paths[frame], 6)
        cold = load_frame(cold_paths[frame], 6)
        bh = load_frame(bh_paths[frame], 3)

        hot_pos = hot[:, 0:3]
        cold_pos = cold[:, 0:3]
        hot_rho = hot[:, 3]
        cold_rho = cold[:, 3]
        hot_phi = hot[:, 4]
        cold_phi = cold[:, 4]
        hot_int = hot[:, 5]
        cold_int = cold[:, 5]

        combined_pos = np.vstack([hot_pos, cold_pos])
        combined_rho = np.hstack([hot_rho, cold_rho])
        combined_phi = np.hstack([hot_phi, cold_phi])
        combined_int = np.hstack([hot_int, cold_int])

        rho_norm_hot = normalize_field(hot_rho, "linear") ** args.rho_gamma
        rho_norm_cold = normalize_field(cold_rho, "linear") ** args.rho_gamma
        rho_norm_combined = normalize_field(combined_rho, "linear") ** args.rho_gamma

        phi_norm_hot = normalize_field(hot_phi, args.phi_scale) ** args.phi_gamma
        phi_norm_cold = normalize_field(cold_phi, args.phi_scale) ** args.phi_gamma
        phi_norm_combined = normalize_field(combined_phi, args.phi_scale) ** args.phi_gamma

        colors_top = mix_colors(rho_norm_combined, phi_norm_combined, rho_color, phi_color, combined_int)
        colors_hot = mix_colors(rho_norm_hot, phi_norm_hot, rho_color, phi_color, hot_int)
        colors_cold = mix_colors(rho_norm_cold, phi_norm_cold, rho_color, phi_color, cold_int)

        scatter_top._offsets3d = (combined_pos[:, 0], combined_pos[:, 1], combined_pos[:, 2])
        scatter_hot._offsets3d = (hot_pos[:, 0], hot_pos[:, 1], hot_pos[:, 2])
        scatter_cold._offsets3d = (cold_pos[:, 0], cold_pos[:, 1], cold_pos[:, 2])

        scatter_top.set_facecolor(colors_top)
        scatter_hot.set_facecolor(colors_hot)
        scatter_cold.set_facecolor(colors_cold)

        bh_core_top._offsets3d = (bh[:, 0], bh[:, 1], bh[:, 2])
        bh_halo_top._offsets3d = (bh[:, 0], bh[:, 1], bh[:, 2])
        bh_core_hot._offsets3d = (bh[:, 0], bh[:, 1], bh[:, 2])
        bh_core_cold._offsets3d = (bh[:, 0], bh[:, 1], bh[:, 2])

        spin = frame * 0.02
        cs = np.cos(spin)
        sn = np.sin(spin)
        x = smbh_loop[0] * cs - smbh_loop[1] * sn
        y = smbh_loop[0] * sn + smbh_loop[1] * cs
        z = smbh_loop[2]
        for line in smbh_lines:
            update_line(line, x, y, z)

        if not args.no_bh_lines:
            for i, line in enumerate(bh_lines):
                update_line(line, bh_loop[0] + bh[i, 0], bh_loop[1] + bh[i, 1], bh_loop[2] + bh[i, 2])

        return scatter_top, scatter_hot, scatter_cold

    ani = animation.FuncAnimation(fig, update, frames=frame_count, interval=1000 / args.fps)
    writer = animation.FFMpegWriter(fps=args.fps, bitrate=2400)
    ani.save(args.out, writer=writer, dpi=120)
    print(f"Saved {args.out}")


if __name__ == "__main__":
    main()
