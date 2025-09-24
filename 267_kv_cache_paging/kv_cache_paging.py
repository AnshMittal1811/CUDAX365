import argparse
import os
import time

import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--size", type=int, default=1_000_000)
    parser.add_argument("--cache", default="kv_cache.dat")
    args = parser.parse_args()

    if not os.path.exists(args.cache):
        mm = np.memmap(args.cache, dtype=np.float32, mode="w+", shape=(args.size,))
        mm[:] = np.random.rand(args.size).astype(np.float32)
        mm.flush()

    mm = np.memmap(args.cache, dtype=np.float32, mode="r", shape=(args.size,))

    start = time.time()
    sample = float(mm[args.size // 2])
    elapsed = (time.time() - start) * 1000.0

    with open("kv_cache_log.txt", "w", encoding="utf-8") as f:
        f.write(f"sample={sample:.6f}\n")
        f.write(f"read_ms={elapsed:.3f}\n")

    print("Wrote kv_cache_log.txt")


if __name__ == "__main__":
    main()
