import subprocess
import numpy as np

actions = [0.05, 0.1, 0.2, 0.3, 0.4]
Q = np.zeros((3, len(actions)))  # states: stable/ok/unstable
alpha, gamma, eps = 0.3, 0.9, 0.2


def run_sim(cfl):
    p = subprocess.run(
        ["./mhd_qlearn_cfl", "128", "128", "50", "--cfl", str(cfl)],
        capture_output=True,
        text=True,
        check=False,
    )
    out = p.stdout
    drift = 0.0
    nan = 0
    for line in out.splitlines():
        if "Energy drift:" in line:
            drift = float(line.split()[-1])
        if "NaN:" in line:
            nan = int(line.split()[-1])
    reward = -abs(drift) - (10.0 if nan else 0.0)
    if nan:
        state = 2
    elif abs(drift) > 1e-2:
        state = 1
    else:
        state = 0
    return state, reward


def main():
    for episode in range(50):
        state = 0
        for _ in range(10):
            if np.random.rand() < eps:
                a = np.random.randint(len(actions))
            else:
                a = int(np.argmax(Q[state]))
            next_state, reward = run_sim(actions[a])
            Q[state, a] += alpha * (reward + gamma * np.max(Q[next_state]) - Q[state, a])
            state = next_state
        print("episode", episode, "Q", Q)


if __name__ == "__main__":
    main()
