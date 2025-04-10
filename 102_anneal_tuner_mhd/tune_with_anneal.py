import subprocess
import numpy as np


def run_gpu_anneal():
    try:
        out = subprocess.check_output(["../101_qi_anneal_ptx/anneal_ptx"], text=True)
        return out.strip()
    except Exception:
        return None


def main():
    result = run_gpu_anneal()
    print("anneal result", result)

    # simple hyperparam grid using mock data
    rho = np.fromfile("rho.bin", dtype=np.float32)
    phi = np.fromfile("phi.bin", dtype=np.float32)
    grad = np.abs(np.gradient(rho.reshape(128, 128))[0]).mean()

    best = None
    for dt in np.linspace(0.8, 1.2, 9):
        for ps in np.linspace(0.05, 0.2, 7):
            score = abs(dt - 1.0) + abs(ps - 0.1) + 0.1 * grad
            if best is None or score < best[0]:
                best = (score, dt, ps)
    print("best hyperparams", best)


if __name__ == "__main__":
    main()
