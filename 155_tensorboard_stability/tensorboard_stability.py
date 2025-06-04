import argparse
import csv
import os


def load_rewards(path):
    rewards = []
    with open(path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rewards.append(float(row["reward"]))
    return rewards


def write_csv(path, rewards):
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["episode", "reward"])
        for i, r in enumerate(rewards):
            writer.writerow([i, r])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", default="../154_rl_random_resets/rewards_random_resets.csv")
    parser.add_argument("--random", default="../154_rl_random_resets/rewards_random_resets.csv")
    parser.add_argument("--logdir", default="runs")
    args = parser.parse_args()

    if not os.path.exists(args.baseline):
        print("Baseline rewards missing; generating mock data")
        write_csv(args.baseline, [100.0] * 200)

    if not os.path.exists(args.random):
        print("Random rewards missing; generating mock data")
        write_csv(args.random, [80.0] * 200)

    baseline = load_rewards(args.baseline)
    random_rewards = load_rewards(args.random)

    try:
        from torch.utils.tensorboard import SummaryWriter
    except Exception:
        print("TensorBoard not available; writing CSV summary")
        write_csv("tensorboard_baseline.csv", baseline)
        write_csv("tensorboard_random.csv", random_rewards)
        return

    writer = SummaryWriter(args.logdir)
    for i, r in enumerate(baseline):
        writer.add_scalar("reward/baseline", r, i)
    for i, r in enumerate(random_rewards):
        writer.add_scalar("reward/random", r, i)
    writer.close()
    print(f"Wrote TensorBoard logs to {args.logdir}")


if __name__ == "__main__":
    main()
