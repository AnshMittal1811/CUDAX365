import argparse
import time

import numpy as np


def process_chunk(x):
    return x * 1.001 + 0.1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--size", type=int, default=1_000_000)
    parser.add_argument("--out", default="chunk_tuning.csv")
    args = parser.parse_args()

    data = np.random.rand(args.size).astype(np.float32)
    chunk_sizes = [1024, 4096, 16384, 65536, 262144]

    with open(args.out, "w", encoding="utf-8") as f:
        f.write("chunk_size,ms\n")
        for chunk in chunk_sizes:
            start = time.time()
            for i in range(0, args.size, chunk):
                _ = process_chunk(data[i : i + chunk])
            elapsed = (time.time() - start) * 1000.0
            f.write(f"{chunk},{elapsed:.4f}\n")

    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
