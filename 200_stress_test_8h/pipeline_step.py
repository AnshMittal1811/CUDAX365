import time
import numpy as np


def run_step(size=512):
    data = np.random.rand(size, size).astype(np.float32)
    data = data * 1.001 + 0.1
    return float(data.mean())


def main():
    start = time.time()
    value = run_step()
    elapsed = time.time() - start
    print(f"step_value={value:.6f} elapsed={elapsed:.4f}s")


if __name__ == "__main__":
    main()
