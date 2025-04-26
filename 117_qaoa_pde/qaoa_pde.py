import numpy as np
import pennylane as qml

n_qubits = 4

def get_device():
    try:
        return qml.device("lightning.gpu", wires=n_qubits)
    except Exception:
        return qml.device("default.qubit", wires=n_qubits)


def main():
    dev = get_device()

    @qml.qnode(dev)
    def circuit(theta):
        for i in range(n_qubits):
            qml.RY(theta[i], wires=i)
        for i in range(n_qubits - 1):
            qml.CNOT(wires=[i, i + 1])
        return qml.expval(qml.PauliZ(0))

    theta = np.random.randn(n_qubits)
    energy = circuit(theta)

    u = np.random.randn(64, 64)
    lap = np.roll(u, 1, 0) + np.roll(u, -1, 0) + np.roll(u, 1, 1) + np.roll(u, -1, 1) - 4 * u
    u_next = u + 0.05 * float(energy) * lap
    print("energy", energy, "mean", u_next.mean())


if __name__ == "__main__":
    main()
