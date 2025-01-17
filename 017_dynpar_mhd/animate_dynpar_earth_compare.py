import argparse
import re
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import animation, cm, colors
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401


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


def rot_y(angle_rad):
    c = np.cos(angle_rad)
    s = np.sin(angle_rad)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])


def rot_z(angle_rad):
    c = np.cos(angle_rad)
    s = np.sin(angle_rad)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def base_dipole_lines(n_shells, n_phi, n_theta, line_scale, l_min, l_max):
    theta = np.linspace(np.deg2rad(6.0), np.pi - np.deg2rad(6.0), n_theta)
    shells = np.linspace(l_min, l_max, n_shells)
    lines = []
    for L in shells:
        r = line_scale * L * (np.sin(theta) ** 2)
        for phi in np.linspace(0.0, 2.0 * np.pi, n_phi, endpoint=False):
            x = r * np.sin(theta) * np.cos(phi)
            y = r * np.sin(theta) * np.sin(phi)
            z = r * np.cos(theta)
            lines.append(np.vstack([x, y, z]))
    return lines


def rotate_lines(lines, tilt_deg, spin_deg, center):
    tilt = np.deg2rad(tilt_deg)
    spin = np.deg2rad(spin_deg)
    rmat = rot_z(spin) @ rot_y(tilt)
    out = []
    for pts in lines:
        rot = rmat @ pts
        rot[0] += center[0]
        rot[1] += center[1]
        rot[2] += center[2]
        out.append(rot)
    return out


def draw_earth(ax, center, radius):
    u = np.linspace(0.0, 2.0 * np.pi, 40)
    v = np.linspace(0.0, np.pi, 20)
    x = center[0] + radius * np.outer(np.cos(u), np.sin(v))
    y = center[1] + radius * np.outer(np.sin(u), np.sin(v))
    z = center[2] + radius * np.outer(np.ones_like(u), np.cos(v))
    ax.plot_surface(x, y, z, color="#1a2336", alpha=0.35, linewidth=0.0, shade=True)


def make_surface(ax, X, Y, Z, phi, cmap_phi, norm_phi):
    colors_phi = cmap_phi(norm_phi(phi))
    return ax.plot_surface(
        X, Y, Z,
        facecolors=colors_phi,
        rstride=1, cstride=1,
        linewidth=0.0, antialiased=False, shade=False
    )


def main():
    ap = argparse.ArgumentParser(
        description="Compare dynpar vs 014_5 with 3D rho/phi + Earth dipole lines."
    )
    ap.add_argument("--dynpar-rho", default="frames_refine/rho_*.bin")
    ap.add_argument("--dynpar-phi", default="frames_refine/phi_*.bin")
    ap.add_argument("--dynpar-bx", default="frames_refine/bx_*.bin")
    ap.add_argument("--dynpar-by", default="frames_refine/by_*.bin")
    ap.add_argument("--baseline-rho", default="../014_5_advanced_mhd_fp16_tensorcore/frames/rho_*.bin")
    ap.add_argument("--baseline-phi", default="../014_5_advanced_mhd_fp16_tensorcore/frames/phi_*.bin")
    ap.add_argument("--dynpar-shape", nargs=2, type=int, default=[128, 128], metavar=("NY", "NX"))
    ap.add_argument("--baseline-shape", nargs=2, type=int, default=[192, 192], metavar=("NY", "NX"))
    ap.add_argument("--out", default="dynpar_earth_compare_3d.mp4")
    ap.add_argument("--fps", type=int, default=12)
    ap.add_argument("--dt", type=float, default=1.0)
    ap.add_argument("--stride", type=int, default=2)
    ap.add_argument("--rho-scale", type=float, default=1.0)
    ap.add_argument("--plane-z", type=float, default=0.0)
    ap.add_argument("--earth-radius", type=float, default=0.08)
    ap.add_argument("--line-scale", type=float, default=0.42)
    ap.add_argument("--line-shells", type=int, default=5)
    ap.add_argument("--line-phis", type=int, default=8)
    ap.add_argument("--line-steps", type=int, default=120)
    ap.add_argument("--tilt-deg", type=float, default=20.0)
    ap.add_argument("--spin-deg", type=float, default=2.5)
    ap.add_argument("--title", default="DynPar vs 014_5 with Earth B-lines")
    args = ap.parse_args()

    dyn_rho, dyn_labels = load_frames(args.dynpar_rho, args.dynpar_shape)
    dyn_phi, _ = load_frames(args.dynpar_phi, args.dynpar_shape)
    base_rho, base_labels = load_frames(args.baseline_rho, args.baseline_shape)
    base_phi, _ = load_frames(args.baseline_phi, args.baseline_shape)

    dyn_bx = dyn_by = None
    try:
        dyn_bx, _ = load_frames(args.dynpar_bx, args.dynpar_shape)
        dyn_by, _ = load_frames(args.dynpar_by, args.dynpar_shape)
    except SystemExit:
        dyn_bx = dyn_by = None

    n = min(len(dyn_rho), len(base_rho))
    dyn_rho = dyn_rho[:n]
    dyn_phi = dyn_phi[:n]
    base_rho = base_rho[:n]
    base_phi = base_phi[:n]
    dyn_labels = dyn_labels[:n]
    base_labels = base_labels[:n]
    if dyn_bx is not None:
        dyn_bx = dyn_bx[:n]
        dyn_by = dyn_by[:n]

    rho_min = min(min(f.min() for f in dyn_rho), min(f.min() for f in base_rho))
    rho_max = max(max(f.max() for f in dyn_rho), max(f.max() for f in base_rho))
    phi_min = min(min(f.min() for f in dyn_phi), min(f.min() for f in base_phi))
    phi_max = max(max(f.max() for f in dyn_phi), max(f.max() for f in base_phi))

    cmap_phi = cm.magma
    norm_phi = colors.Normalize(vmin=phi_min, vmax=phi_max)

    stride = max(1, args.stride)
    dyn_ny, dyn_nx = args.dynpar_shape
    base_ny, base_nx = args.baseline_shape
    xd = np.linspace(0.0, 1.0, dyn_nx)[::stride]
    yd = np.linspace(0.0, 1.0, dyn_ny)[::stride]
    xb = np.linspace(0.0, 1.0, base_nx)[::stride]
    yb = np.linspace(0.0, 1.0, base_ny)[::stride]
    XD, YD = np.meshgrid(xd, yd)
    XB, YB = np.meshgrid(xb, yb)

    line_extent = args.line_scale * 1.0
    rho_extent = (rho_max - rho_min) * args.rho_scale
    zmin = min(args.plane_z - line_extent, args.plane_z)
    zmax = max(args.plane_z + line_extent, args.plane_z + rho_extent)

    fig = plt.figure(figsize=(12, 5))
    ax1 = fig.add_subplot(1, 2, 1, projection="3d")
    ax2 = fig.add_subplot(1, 2, 2, projection="3d")

    for ax in (ax1, ax2):
        ax.set_xlim(0.0, 1.0)
        ax.set_ylim(0.0, 1.0)
        ax.set_zlim(zmin, zmax)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("rho")
        ax.view_init(elev=28.0, azim=-60.0)
        ax.set_box_aspect((1.0, 1.0, 0.6))

    ax1.set_title("dynpar (refine)")
    ax2.set_title("014_5 baseline")

    sm = cm.ScalarMappable(norm=norm_phi, cmap=cmap_phi)
    sm.set_array([])
    fig.colorbar(sm, ax=[ax1, ax2], shrink=0.6, pad=0.08, label="phi")

    earth_center = (0.5, 0.5, args.plane_z)
    draw_earth(ax1, earth_center, args.earth_radius)
    draw_earth(ax2, earth_center, args.earth_radius)

    base_lines = base_dipole_lines(
        args.line_shells, args.line_phis, args.line_steps,
        args.line_scale, 0.6, 1.0
    )
    line_objs_1 = []
    line_objs_2 = []
    lines0 = rotate_lines(base_lines, args.tilt_deg, 0.0, earth_center)
    for pts in lines0:
        line, = ax1.plot(pts[0], pts[1], pts[2], color="#34a1c7", lw=0.8, alpha=0.6)
        line_objs_1.append(line)
        line, = ax2.plot(pts[0], pts[1], pts[2], color="#34a1c7", lw=0.8, alpha=0.6)
        line_objs_2.append(line)

    surf1 = [None]
    surf2 = [None]

    if dyn_bx is not None:
        bmag = [float(np.sqrt(bx * bx + by * by).mean()) for bx, by in zip(dyn_bx, dyn_by)]
        bmin = min(bmag)
        bmax = max(bmag)
    else:
        bmag = None
        bmin = bmax = 0.0

    def update(i):
        if surf1[0] is not None:
            surf1[0].remove()
        if surf2[0] is not None:
            surf2[0].remove()

        rho_d = (dyn_rho[i] - rho_min) * args.rho_scale + args.plane_z
        phi_d = dyn_phi[i]
        rho_b = (base_rho[i] - rho_min) * args.rho_scale + args.plane_z
        phi_b = base_phi[i]

        surf1[0] = make_surface(
            ax1, XD, YD,
            rho_d[::stride, ::stride],
            phi_d[::stride, ::stride],
            cmap_phi, norm_phi
        )
        surf2[0] = make_surface(
            ax2, XB, YB,
            rho_b[::stride, ::stride],
            phi_b[::stride, ::stride],
            cmap_phi, norm_phi
        )

        spin = args.spin_deg * i
        lines = rotate_lines(base_lines, args.tilt_deg, spin, earth_center)

        if bmag is not None and bmax > bmin:
            alpha = 0.25 + 0.7 * (bmag[i] - bmin) / (bmax - bmin)
        else:
            alpha = 0.6

        for line_obj, pts in zip(line_objs_1, lines):
            line_obj.set_data(pts[0], pts[1])
            line_obj.set_3d_properties(pts[2])
            line_obj.set_alpha(alpha)
        for line_obj, pts in zip(line_objs_2, lines):
            line_obj.set_data(pts[0], pts[1])
            line_obj.set_3d_properties(pts[2])
            line_obj.set_alpha(alpha)

        t = label_time(dyn_labels[i], i, args.dt)
        fig.suptitle(f"{args.title}  t={t:.2f}")
        ax1.set_title(f"dynpar (refine)  {dyn_labels[i]}")
        ax2.set_title(f"014_5 baseline  {base_labels[i]}")
        return []

    ani = animation.FuncAnimation(fig, update, frames=n, interval=1000 / args.fps, blit=False)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    writer = "pillow" if out_path.suffix.lower() == ".gif" else "ffmpeg"
    ani.save(out_path, writer=writer, dpi=120)
    plt.close(fig)
    print(f"Saved animation to {out_path} using {n} frames.")


if __name__ == "__main__":
    main()
