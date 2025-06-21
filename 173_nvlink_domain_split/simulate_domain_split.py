import argparse
import time

import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--nx", type=int, default=256)
    parser.add_argument("--ny", type=int, default=256)
    parser.add_argument("--steps", type=int, default=50)
    args = parser.parse_args()

    nx, ny = args.nx, args.ny
    left = np.zeros((ny, nx // 2), dtype=np.float32)
    right = np.zeros((ny, nx // 2), dtype=np.float32)

    start = time.time()
    for _ in range(args.steps):
        left[:, -1] = right[:, 0]
        right[:, 0] = left[:, -1]
        left[1:-1, 1:-1] = 0.25 * (
            left[1:-1, :-2] + left[1:-1, 2:] + left[:-2, 1:-1] + left[2:, 1:-1]
        )
        right[1:-1, 1:-1] = 0.25 * (
            right[1:-1, :-2] + right[1:-1, 2:] + right[:-2, 1:-1] + right[2:, 1:-1]
        )
    elapsed = time.time() - start

    with open("domain_split_log.txt", "w", encoding="utf-8") as f:
        f.write(f"steps={args.steps}\n")
        f.write(f"elapsed_sec={elapsed:.4f}\n")

    print("Wrote domain_split_log.txt")


if __name__ == "__main__":
    main()
