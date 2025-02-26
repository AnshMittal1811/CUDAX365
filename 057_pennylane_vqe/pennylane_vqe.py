import pennylane as qml
import numpy as np

n_qubits = 4

dev = qml.device("lightning.gpu", wires=n_qubits)

@qml.qnode(dev)
def circuit(theta):
    for i in range(n_qubits):
        qml.RY(theta[i], wires=i)
    for i in range(n_qubits-1):
        qml.CNOT(wires=[i, i+1])
    return qml.expval(qml.PauliZ(0) @ qml.PauliZ(1))


def main():
    theta = np.random.randn(n_qubits)
    energy = circuit(theta)
    print("energy", energy)


if __name__ == "__main__":
    main()
