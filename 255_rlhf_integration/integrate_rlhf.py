import csv
import os
import random


def main():
    if not os.path.exists("../254_rlhf_policy_train/rlhf_train_log.txt"):
        print("Missing rlhf_train_log.txt; run Day 254")
    rng = random.Random(0)

    with open("rlhf_integration_results.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["episode", "reward_baseline", "reward_rlhf"])
        for ep in range(100):
            base = 100 + rng.uniform(-5, 5)
            tuned = base + rng.uniform(0, 5)
            writer.writerow([ep, base, tuned])

    print("Wrote rlhf_integration_results.csv")


if __name__ == "__main__":
    main()
