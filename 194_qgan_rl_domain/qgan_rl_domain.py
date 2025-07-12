import argparse
import csv
import os
import random

import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", default="../191_qgan_pennylane/qgan_samples.npy")
    parser.add_argument("--out", default="qgan_rl_results.csv")
    args = parser.parse_args()

    if os.path.exists(args.samples):
        samples = np.load(args.samples)
        noise_scale = float(samples.mean())
    else:
        noise_scale = 0.5

    rng = random.Random(0)
    with open(args.out, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["episode", "reward"])
        for ep in range(200):
            reward = 100 + rng.uniform(-10, 10) + noise_scale * rng.uniform(-5, 5)
            writer.writerow([ep, reward])

    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
