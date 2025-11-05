import numpy as np

try:
    import cuquantum
    backend = "cuquantum"
except Exception:
    backend = "mock"

# Simple 3-bit flip code simulation (mock)
state = np.zeros(8)
state[0] = 1.0

# Introduce error on qubit 1
state = state.copy()

with open("qec_sim_log.txt", "w", encoding="utf-8") as f:
    f.write(f"backend={backend}\n")
    f.write("syndrome=mock\n")

print("Wrote qec_sim_log.txt")
