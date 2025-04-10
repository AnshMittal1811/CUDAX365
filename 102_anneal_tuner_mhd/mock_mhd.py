import numpy as np


def score_hyperparams(dt_scale, phi_scale):
    rho = np.fromfile("rho.bin", dtype=np.float32)
    phi = np.fromfile("phi.bin", dtype=np.float32)
    # toy stability score: penalize large gradients and too large dt
    grad = np.abs(np.gradient(rho.reshape(128, 128))[0]).mean()
    return abs(dt_scale - 1.0) + abs(phi_scale - 0.1) + 0.1 * grad


def main():
    best = None
    for dt in [0.8, 1.0, 1.2]:
        for ps in [0.05, 0.1, 0.2]:
            s = score_hyperparams(dt, ps)
            if best is None or s < best[0]:
                best = (s, dt, ps)
    print("best score", best)


if __name__ == "__main__":
    main()
