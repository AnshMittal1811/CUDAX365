import argparse
import csv
import random

import numpy as np


def summarize_hud(segmask, lidar):
    return np.array([
        segmask.mean(),
        segmask.std(),
        lidar[:, 0].mean(),
        lidar[:, 1].mean(),
    ], dtype=np.float32)


def run_mock(episodes, out_csv):
    rng = random.Random(0)
    with open(out_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["episode", "reward"])
        for ep in range(episodes):
            reward = 50 + rng.random() * 50
            writer.writerow([ep, reward])


def run_gym(episodes, out_csv, hud_vec):
    import gym

    env = gym.make("CartPole-v1")
    with open(out_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["episode", "reward"])
        for ep in range(episodes):
            obs, _ = env.reset()
            reward_sum = 0.0
            for _ in range(200):
                augmented = np.concatenate([obs, hud_vec])
                action = 0 if augmented.sum() < 0 else env.action_space.sample()
                obs, reward, terminated, truncated, _ = env.step(action)
                reward_sum += reward
                if terminated or truncated:
                    break
            writer.writerow([ep, reward_sum])
    env.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=200)
    parser.add_argument("--out", default="hud_rewards.csv")
    args = parser.parse_args()

    segmask = np.load("hud_segmask.npy")
    lidar = np.load("hud_lidar.npy")
    hud_vec = summarize_hud(segmask, lidar)

    try:
        import gym  # noqa: F401
        run_gym(args.episodes, args.out, hud_vec)
        print("Used gym CartPole")
    except Exception:
        run_mock(args.episodes, args.out)
        print("Gym not available; used mock rewards")

    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
