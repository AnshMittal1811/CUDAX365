import csv
import os


def load_rewards(path):
    rewards = []
    with open(path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rewards.append(float(row["reward"]))
    return rewards


def success_rate(rewards, threshold=150.0):
    if not rewards:
        return 0.0
    success = sum(1 for r in rewards if r >= threshold)
    return success / len(rewards)


def main():
    baseline_path = "../154_rl_random_resets/rewards_random_resets.csv"
    hud_path = "../185_rl_hud_fusion/hud_rewards.csv"

    if not os.path.exists(baseline_path) or not os.path.exists(hud_path):
        raise SystemExit("Missing reward files; run Day 154 and 185 first")

    baseline = load_rewards(baseline_path)
    hud = load_rewards(hud_path)

    rate_base = success_rate(baseline)
    rate_hud = success_rate(hud)

    with open("success_rate.txt", "w", encoding="utf-8") as f:
        f.write(f"baseline_rate={rate_base:.3f}\n")
        f.write(f"hud_rate={rate_hud:.3f}\n")

    print("Wrote success_rate.txt")


if __name__ == "__main__":
    main()
