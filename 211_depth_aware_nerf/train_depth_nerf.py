import argparse
import time
import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--images", default="images.npy")
    parser.add_argument("--depths", default="depths.npy")
    parser.add_argument("--out", default="depth_nerf_log.txt")
    args = parser.parse_args()

    images = np.load(args.images)
    depths = np.load(args.depths)

    start = time.time()
    loss = float(np.mean((images[..., 0] - depths) ** 2))
    elapsed = time.time() - start

    with open(args.out, "w", encoding="utf-8") as f:
        f.write(f"loss={loss:.6f}\n")
        f.write(f"elapsed_sec={elapsed:.4f}\n")

    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
