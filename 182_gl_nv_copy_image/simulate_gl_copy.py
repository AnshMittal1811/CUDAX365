import argparse
import time

import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--size", type=int, default=1024)
    parser.add_argument("--iters", type=int, default=100)
    args = parser.parse_args()

    img = np.random.rand(args.size, args.size, 4).astype(np.float32)

    start = time.time()
    for _ in range(args.iters):
        direct = img.copy()
    direct_ms = (time.time() - start) * 1000.0

    start = time.time()
    for _ in range(args.iters):
        staging = np.empty_like(img)
        staging[:] = img
        direct = staging.copy()
    staging_ms = (time.time() - start) * 1000.0

    with open("copy_times.csv", "w", encoding="utf-8") as f:
        f.write("mode,ms\n")
        f.write(f"direct,{direct_ms:.4f}\n")
        f.write(f"staging,{staging_ms:.4f}\n")

    print("Wrote copy_times.csv")


if __name__ == "__main__":
    main()
