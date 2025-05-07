import numpy as np

try:
    import gymnasium as gym
except Exception:
    gym = None


def main():
    if gym is None:
        print("gymnasium not available; using mock env")
        obs = np.random.randn(100, 4)
        noise = 0.1 * np.random.randn(*obs.shape)
        noisy = obs + noise
        print("mock stability", np.std(noisy))
        return

    env = gym.make("CartPole-v1")
    obs, _ = env.reset()
    total = 0.0
    for _ in range(200):
        noise = 0.05 * np.random.randn(*obs.shape)
        noisy_obs = obs + noise
        action = 0 if noisy_obs[2] < 0 else 1
        obs, reward, done, _, _ = env.step(action)
        total += reward
        if done:
            obs, _ = env.reset()
    print("total reward", total)


if __name__ == "__main__":
    main()
