import argparse
import numpy as np


def fid(real, fake):
    mu1 = real.mean(axis=0)
    mu2 = fake.mean(axis=0)
    cov1 = np.cov(real, rowvar=False)
    cov2 = np.cov(fake, rowvar=False)

    diff = mu1 - mu2
    diff_sq = diff.dot(diff)

    covmean = cov1 @ cov2
    trace = np.trace(cov1 + cov2 - 2 * np.sqrt(np.abs(covmean)))
    return diff_sq + trace


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--real", default="real_features.npy")
    parser.add_argument("--fake", default="fake_features.npy")
    parser.add_argument("--out", default="fid_score.txt")
    args = parser.parse_args()

    real = np.load(args.real)
    fake = np.load(args.fake)
    score = fid(real, fake)

    with open(args.out, "w", encoding="utf-8") as f:
        f.write(f"fid={score:.4f}\n")

    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
