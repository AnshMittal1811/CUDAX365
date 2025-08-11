import argparse
import time

import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--size", type=int, default=8_000_000)
    parser.add_argument("--out", default="chunk_copy.csv")
    args = parser.parse_args()

    data = np.random.rand(args.size).astype(np.float32)
    chunk_sizes = [1024, 4096, 16384, 65536, 262144, 1048576]

    with open(args.out, "w", encoding="utf-8") as f:
        f.write("chunk,ms\n")
        for chunk in chunk_sizes:
            start = time.time()
            for i in range(0, args.size, chunk):
                _ = data[i : i + chunk].copy()
            elapsed = (time.time() - start) * 1000.0
            f.write(f"{chunk},{elapsed:.4f}\n")

    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
