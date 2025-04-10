import time
import numpy as np
import torch


def pde_step(u):
    lap = np.roll(u, 1, 0) + np.roll(u, -1, 0) + np.roll(u, 1, 1) + np.roll(u, -1, 1) - 4 * u
    return u + 0.1 * lap


def main():
    # PDE loop
    u = np.random.randn(128, 128).astype(np.float32)
    for _ in range(50):
        u = pde_step(u)

    # ML step
    x = torch.randn(1024, 1024, device="cuda")
    for _ in range(10):
        x = x @ x
    torch.cuda.synchronize()

    # Render step (mock)
    img = (u - u.min()) / (u.max() - u.min() + 1e-6)
    img = (img * 255).astype(np.uint8)
    img.tofile("render.bin")
    print("pipeline done")


if __name__ == "__main__":
    main()
