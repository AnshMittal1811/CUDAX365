import argparse
import csv
import random

import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--depths", default="../211_depth_aware_nerf/depths.npy")
    parser.add_argument("--out", default="depth_penalty_rewards.csv")
    args = parser.parse_args()

    depths = np.load(args.depths)
    target = depths.mean()

    rng = random.Random(0)
    with open(args.out, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["episode", "reward"])
        for ep in range(200):
            pred = target + rng.uniform(-0.2, 0.2)
            penalty = abs(pred - target)
            reward = 100.0 - penalty * 50.0
            writer.writerow([ep, reward])

    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
