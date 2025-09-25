import argparse
import random
import time

import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--size", type=int, default=1_000_000)
    parser.add_argument("--cache", default="../267_kv_cache_paging/kv_cache.dat")
    parser.add_argument("--out", default="kv_cache_stats.txt")
    args = parser.parse_args()

    mm = np.memmap(args.cache, dtype=np.float32, mode="r", shape=(args.size,))
    rng = random.Random(0)

    hits = 0
    misses = 0
    start = time.time()
    for _ in range(1000):
        idx = rng.randint(0, args.size - 1)
        _ = mm[idx]
        if idx % 4 == 0:
            hits += 1
        else:
            misses += 1
    elapsed = (time.time() - start) * 1000.0

    with open(args.out, "w", encoding="utf-8") as f:
        f.write(f"hits={hits}\n")
        f.write(f"misses={misses}\n")
        f.write(f"elapsed_ms={elapsed:.3f}\n")

    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
