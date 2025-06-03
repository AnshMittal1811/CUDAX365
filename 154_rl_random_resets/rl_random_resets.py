import argparse
import csv
import random


def run_mock(episodes, max_steps, reset_prob, out_csv):
    rng = random.Random(0)
    with open(out_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["episode", "reward"])
        for ep in range(episodes):
            reward = 0.0
            for _ in range(max_steps):
                reward += rng.uniform(0.0, 1.0)
                if rng.random() < reset_prob:
                    break
            writer.writerow([ep, reward])


def run_gym(episodes, max_steps, reset_prob, out_csv):
    import gym

    rng = random.Random(0)
    env = gym.make("CartPole-v1")
    with open(out_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["episode", "reward"])
        for ep in range(episodes):
            obs, _ = env.reset()
            reward_sum = 0.0
            for _ in range(max_steps):
                action = env.action_space.sample()
                obs, reward, terminated, truncated, _ = env.step(action)
                reward_sum += reward
                if rng.random() < reset_prob:
                    break
                if terminated or truncated:
                    break
            writer.writerow([ep, reward_sum])
    env.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=200)
    parser.add_argument("--max-steps", type=int, default=200)
    parser.add_argument("--reset-prob", type=float, default=0.1)
    parser.add_argument("--out", default="rewards_random_resets.csv")
    args = parser.parse_args()

    try:
        import gym  # noqa: F401
        run_gym(args.episodes, args.max_steps, args.reset_prob, args.out)
        print("Used gym CartPole")
    except Exception:
        run_mock(args.episodes, args.max_steps, args.reset_prob, args.out)
        print("Gym not available; used mock environment")

    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
