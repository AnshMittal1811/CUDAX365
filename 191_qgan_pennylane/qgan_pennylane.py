import argparse
import time
import numpy as np


def run_mock(samples):
    rng = np.random.RandomState(1)
    fake = rng.normal(loc=0.5, scale=0.2, size=samples.shape)
    return fake


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--real", default="real_samples.npy")
    parser.add_argument("--out", default="qgan_samples.npy")
    args = parser.parse_args()

    real = np.load(args.real)

    start = time.time()
    try:
        import pennylane as qml
        dev = qml.device("default.qubit", wires=2)

        @qml.qnode(dev)
        def circuit(theta):
            qml.RY(theta[0], wires=0)
            qml.RY(theta[1], wires=1)
            qml.CNOT(wires=[0, 1])
            return qml.probs(wires=[0, 1])

        theta = np.array([0.1, 0.2])
        probs = circuit(theta)
        fake = np.tile(probs[0], (real.shape[0], 1))
        backend = "pennylane"
    except Exception:
        fake = run_mock(real)
        backend = "mock"

    elapsed = time.time() - start
    np.save(args.out, fake)

    with open("qgan_log.txt", "w", encoding="utf-8") as f:
        f.write(f"backend={backend}\n")
        f.write(f"elapsed_sec={elapsed:.4f}\n")

    print("Wrote qgan_samples.npy and qgan_log.txt")


if __name__ == "__main__":
    main()
