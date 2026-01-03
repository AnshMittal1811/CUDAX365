import argparse
import json
import math
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import animation
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401


def rotation_components(inc, node):
    cosi = np.cos(inc)
    sini = np.sin(inc)
    cosn = np.cos(node)
    sinn = np.sin(node)
    return cosi, sini, cosn, sinn


def orbital_state(params, t):
    gm = params["gm_smbh"]
    a0 = params["a0"]
    e = params["e"]
    phase = params["phase"]
    omega0 = params["omega0"]
    inc = params["inc"]
    node = params["node"]
    a_min = params["a_min"]
    tau = params["tau"]

    a_t = a_min + (a0 - a_min) / (1.0 + t / tau)
    theta = omega0 * t + phase
    r = a_t * (1.0 - e * e) / (1.0 + e * np.cos(theta))

    x = r * np.cos(theta)
    y = r * np.sin(theta)

    cosi, sini, cosn, sinn = rotation_components(inc, node)
    X = cosn * x - sinn * cosi * y
    Y = sinn * x + cosn * cosi * y
    Z = sini * y

    # Tangential direction for Doppler term
    tx = -np.sin(theta)
    ty = np.cos(theta)
    tX = cosn * tx - sinn * cosi * ty
    tY = sinn * tx + cosn * cosi * ty
    tZ = sini * ty

    return np.stack([X, Y, Z], axis=-1), np.stack([tX, tY, tZ], axis=-1), r


def build_orbit_params(n_bh, seed, gm_smbh, a_min, t_end):
    rng = np.random.default_rng(seed)
    a0 = rng.uniform(0.7, 1.4, size=n_bh)
    e = rng.uniform(0.0, 0.35, size=n_bh)
    inc = rng.uniform(0.0, np.deg2rad(35.0), size=n_bh)
    node = rng.uniform(0.0, 2.0 * np.pi, size=n_bh)
    phase = rng.uniform(0.0, 2.0 * np.pi, size=n_bh)
    omega0 = np.sqrt(gm_smbh / (a0 ** 3)) * rng.uniform(0.85, 1.15, size=n_bh)

    params = {
        "gm_smbh": gm_smbh,
        "a0": a0,
        "e": e,
        "inc": inc,
        "node": node,
        "phase": phase,
        "omega0": omega0,
        "a_min": np.full(n_bh, a_min),
        "tau": np.full(n_bh, 0.4 * t_end + 1e-3),
    }
    return params


def sample_disk_points(n_points, r0, sigma, height, seed):
    rng = np.random.default_rng(seed)
    r = rng.normal(loc=r0, scale=sigma, size=n_points)
    r = np.clip(r, 0.05, None)
    theta = rng.uniform(0.0, 2.0 * np.pi, size=n_points)
    z = rng.normal(loc=0.0, scale=height, size=n_points)
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    return np.stack([x, y, z], axis=-1), r, theta


def rotate_local(points, inc, node, spin):
    cos_spin = math.cos(spin)
    sin_spin = math.sin(spin)
    x = points[:, 0] * cos_spin - points[:, 1] * sin_spin
    y = points[:, 0] * sin_spin + points[:, 1] * cos_spin
    z = points[:, 2]

    cosi, sini, cosn, sinn = rotation_components(inc, node)
    y1 = y * cosi - z * sini
    z1 = y * sini + z * cosi
    X = cosn * x - sinn * y1
    Y = sinn * x + cosn * y1
    Z = z1
    return np.stack([X, Y, Z], axis=-1)


def compute_rho(r, theta, params, t):
    rho0 = params["rho0"]
    r0 = params["r0"]
    sigma = params["sigma"]
    arm_amp = params["arm_amp"]
    arm_m = params["arm_m"]
    omega = params["omega"]
    profile = rho0 * np.exp(-((r - r0) ** 2) / (2.0 * sigma * sigma))
    spiral = 1.0 + arm_amp * np.sin(arm_m * (theta - omega * t))
    return profile * spiral


def compute_phi(pos, r_local, params):
    gm_smbh = params["gm_smbh"]
    gm_bh = params["gm_bh"]
    rs_smbh = params["rs_smbh"]
    rs_bh = params["rs_bh"]
    r_smbh = np.linalg.norm(pos, axis=1)
    r_smbh = np.maximum(r_smbh, rs_smbh + 1e-3)
    r_local = np.maximum(r_local, rs_bh + 1e-3)
    phi = -gm_smbh / (r_smbh - rs_smbh) - gm_bh / (r_local - rs_bh)
    return phi


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


def relativistic_intensity(r_smbh, v_dir, gm_smbh, rs_smbh, obs_dir):
    r_smbh = np.maximum(r_smbh, rs_smbh + 1e-3)
    v_mag = np.sqrt(gm_smbh / r_smbh)
    beta = np.clip(v_mag, 0.0, 0.6)
    gamma = 1.0 / np.sqrt(1.0 - beta * beta)
    mu = np.clip(np.dot(v_dir, obs_dir), -0.9, 0.9)
    doppler = 1.0 / (gamma * (1.0 - beta * mu))
    grav = np.sqrt(np.maximum(1.0 - rs_smbh / r_smbh, 0.15))
    intensity = doppler * grav
    return np.clip(intensity, 0.6, 1.5)


def main():
    ap = argparse.ArgumentParser(description="Simulate 32 BH accretion around a SMBH")
    ap.add_argument("--n-bh", type=int, default=8)
    ap.add_argument("--frames", type=int, default=1200)
    ap.add_argument("--fps", type=int, default=20)
    ap.add_argument("--dt", type=float, default=0.05)
    ap.add_argument("--points-per-bh", type=int, default=128)
    ap.add_argument("--seed", type=int, default=17)
    ap.add_argument("--out", default="multi_bh_accretion.mp4")
    ap.add_argument("--phi-scale", choices=["linear", "symlog"], default="symlog")
    ap.add_argument("--rho-gamma", type=float, default=1.0)
    ap.add_argument("--phi-gamma", type=float, default=1.0)
    ap.add_argument("--title", default="32-BH Cluster + SMBH Accretion")
    args = ap.parse_args()

    t_end = args.frames * args.dt
    gm_smbh = 30.0
    rs_smbh = 0.25
    gm_bh = 0.6
    rs_bh = 0.04

    orbit_params = build_orbit_params(args.n_bh, args.seed, gm_smbh, a_min=0.35, t_end=t_end)
    orbit_params["gm_smbh"] = gm_smbh

    hot_base, hot_r, hot_theta = sample_disk_points(args.points_per_bh, 0.12, 0.04, 0.02, args.seed + 1)
    cold_base, cold_r, cold_theta = sample_disk_points(args.points_per_bh, 0.18, 0.06, 0.01, args.seed + 2)

    hot_params = {
        "rho0": 1.2,
        "r0": 0.12,
        "sigma": 0.05,
        "arm_amp": 0.4,
        "arm_m": 3.0,
        "omega": 5.0,
    }
    cold_params = {
        "rho0": 0.9,
        "r0": 0.18,
        "sigma": 0.07,
        "arm_amp": 0.25,
        "arm_m": 2.0,
        "omega": 3.5,
    }

    phi_params = {
        "gm_smbh": gm_smbh,
        "gm_bh": gm_bh,
        "rs_smbh": rs_smbh,
        "rs_bh": rs_bh,
    }

    rho_color = np.array([0.2, 0.6, 1.0])
    phi_color = np.array([1.0, 0.45, 0.1])
    obs_dir = np.array([0.0, 0.0, 1.0])

    fig = plt.figure(figsize=(12, 8))
    gs = fig.add_gridspec(2, 2, height_ratios=[2.0, 1.6])
    ax_top = fig.add_subplot(gs[0, :], projection="3d")
    ax_hot = fig.add_subplot(gs[1, 0], projection="3d")
    ax_cold = fig.add_subplot(gs[1, 1], projection="3d")

    for ax in (ax_top, ax_hot, ax_cold):
        ax.set_xlim(-1.6, 1.6)
        ax.set_ylim(-1.6, 1.6)
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

    bh_scatter_top = ax_top.scatter([], [], [], s=12, c="white", depthshade=False)
    bh_scatter_hot = ax_hot.scatter([], [], [], s=12, c="white", depthshade=False)
    bh_scatter_cold = ax_cold.scatter([], [], [], s=12, c="white", depthshade=False)

    def update(frame):
        t = frame * args.dt
        positions, tangents, radii = orbital_state(orbit_params, t)
        intensities = relativistic_intensity(radii, tangents, gm_smbh, rs_smbh, obs_dir)

        hot_pts_all = []
        cold_pts_all = []
        hot_rho_all = []
        cold_rho_all = []
        hot_phi_all = []
        cold_phi_all = []
        hot_int_all = []
        cold_int_all = []

        for i in range(args.n_bh):
            inc = orbit_params["inc"][i]
            node = orbit_params["node"][i]
            spin_hot = hot_params["omega"] * t + i * 0.2
            spin_cold = cold_params["omega"] * t + i * 0.2

            hot_pts = rotate_local(hot_base, inc, node, spin_hot)
            cold_pts = rotate_local(cold_base, inc, node, spin_cold)

            hot_pts += positions[i]
            cold_pts += positions[i]

            hot_rho = compute_rho(hot_r, hot_theta, hot_params, t)
            cold_rho = compute_rho(cold_r, cold_theta, cold_params, t)

            hot_phi = compute_phi(hot_pts, hot_r, phi_params)
            cold_phi = compute_phi(cold_pts, cold_r, phi_params)

            hot_pts_all.append(hot_pts)
            cold_pts_all.append(cold_pts)
            hot_rho_all.append(hot_rho)
            cold_rho_all.append(cold_rho)
            hot_phi_all.append(hot_phi)
            cold_phi_all.append(cold_phi)
            hot_int_all.append(np.full(hot_rho.shape, intensities[i]))
            cold_int_all.append(np.full(cold_rho.shape, intensities[i]))

        hot_pts_all = np.vstack(hot_pts_all)
        cold_pts_all = np.vstack(cold_pts_all)
        hot_rho_all = np.hstack(hot_rho_all)
        cold_rho_all = np.hstack(cold_rho_all)
        hot_phi_all = np.hstack(hot_phi_all)
        cold_phi_all = np.hstack(cold_phi_all)
        hot_int_all = np.hstack(hot_int_all)
        cold_int_all = np.hstack(cold_int_all)

        combined_pts = np.vstack([hot_pts_all, cold_pts_all])
        combined_rho = np.hstack([hot_rho_all, cold_rho_all])
        combined_phi = np.hstack([hot_phi_all, cold_phi_all])
        combined_int = np.hstack([hot_int_all, cold_int_all])

        rho_norm_hot = normalize_field(hot_rho_all, "linear") ** args.rho_gamma
        rho_norm_cold = normalize_field(cold_rho_all, "linear") ** args.rho_gamma
        rho_norm_combined = normalize_field(combined_rho, "linear") ** args.rho_gamma

        phi_norm_hot = normalize_field(hot_phi_all, args.phi_scale) ** args.phi_gamma
        phi_norm_cold = normalize_field(cold_phi_all, args.phi_scale) ** args.phi_gamma
        phi_norm_combined = normalize_field(combined_phi, args.phi_scale) ** args.phi_gamma

        colors_top = mix_colors(rho_norm_combined, phi_norm_combined, rho_color, phi_color, combined_int)
        colors_hot = mix_colors(rho_norm_hot, phi_norm_hot, rho_color, phi_color, hot_int_all)
        colors_cold = mix_colors(rho_norm_cold, phi_norm_cold, rho_color, phi_color, cold_int_all)

        scatter_top._offsets3d = (combined_pts[:, 0], combined_pts[:, 1], combined_pts[:, 2])
        scatter_hot._offsets3d = (hot_pts_all[:, 0], hot_pts_all[:, 1], hot_pts_all[:, 2])
        scatter_cold._offsets3d = (cold_pts_all[:, 0], cold_pts_all[:, 1], cold_pts_all[:, 2])

        scatter_top.set_facecolor(colors_top)
        scatter_hot.set_facecolor(colors_hot)
        scatter_cold.set_facecolor(colors_cold)

        bh_scatter_top._offsets3d = (positions[:, 0], positions[:, 1], positions[:, 2])
        bh_scatter_hot._offsets3d = (positions[:, 0], positions[:, 1], positions[:, 2])
        bh_scatter_cold._offsets3d = (positions[:, 0], positions[:, 1], positions[:, 2])

        return scatter_top, scatter_hot, scatter_cold

    ani = animation.FuncAnimation(fig, update, frames=args.frames, interval=1000 / args.fps)
    writer = animation.FFMpegWriter(fps=args.fps, bitrate=2400)
    ani.save(args.out, writer=writer, dpi=120)
    print(f"Saved {args.out}")

    params_dump = {
        "n_bh": args.n_bh,
        "frames": args.frames,
        "dt": args.dt,
        "gm_smbh": gm_smbh,
        "rs_smbh": rs_smbh,
        "gm_bh": gm_bh,
        "rs_bh": rs_bh,
    }
    Path("params.json").write_text(json.dumps(params_dump, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
