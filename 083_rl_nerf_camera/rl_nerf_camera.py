import numpy as np

angles = np.linspace(0, 2 * np.pi, 8, endpoint=False)
Q = np.zeros((len(angles), len(angles)))
alpha, gamma, eps = 0.3, 0.9, 0.2


def reward(angle):
    return np.cos(angle) + 0.1 * np.random.randn()


def main():
    state = 0
    for ep in range(50):
        for _ in range(10):
            if np.random.rand() < eps:
                action = np.random.randint(len(angles))
            else:
                action = int(np.argmax(Q[state]))
            r = reward(angles[action])
            Q[state, action] += alpha * (r + gamma * np.max(Q[action]) - Q[state, action])
            state = action
        if ep % 10 == 0:
            print("episode", ep, "best angle", angles[np.argmax(Q[0])])


if __name__ == "__main__":
    main()
