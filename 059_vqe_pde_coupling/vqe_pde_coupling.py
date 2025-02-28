import numpy as np
import pennylane as qml

n_qubits = 4

dev = qml.device("lightning.gpu", wires=n_qubits)

@qml.qnode(dev)
def circuit(theta):
    for i in range(n_qubits):
        qml.RY(theta[i], wires=i)
    return qml.expval(qml.PauliZ(0))


def main():
    theta = np.random.randn(n_qubits)
    energy = circuit(theta)
    coef = float(energy)

    u = np.random.randn(64, 64)
    lap = np.roll(u, 1, 0) + np.roll(u, -1, 0) + np.roll(u, 1, 1) + np.roll(u, -1, 1) - 4 * u
    u_next = u + 0.1 * coef * lap
    print("coef", coef, "mean", u_next.mean())


if __name__ == "__main__":
    main()
