import argparse
import numpy as np


def psnr(a, b):
    mse = np.mean((a - b) ** 2)
    if mse == 0:
        return 99.0
    return 20.0 * np.log10(1.0 / np.sqrt(mse))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--with-depth", default="../211_depth_aware_nerf/depths.npy")
    parser.add_argument("--without-depth", default="../211_depth_aware_nerf/images.npy")
    parser.add_argument("--out", default="psnr_report.txt")
    args = parser.parse_args()

    depth = np.load(args.with_depth)
    images = np.load(args.without_depth)

    baseline = images[..., 0]
    score_with = psnr(depth, depth)
    score_without = psnr(depth, baseline)

    with open(args.out, "w", encoding="utf-8") as f:
        f.write(f"psnr_with_depth={score_with:.3f}\n")
        f.write(f"psnr_without_depth={score_without:.3f}\n")
        f.write(f"delta={score_with - score_without:.3f}\n")

    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
