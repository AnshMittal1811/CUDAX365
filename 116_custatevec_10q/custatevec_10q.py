import time
import numpy as np

try:
    from cuquantum import custatevec
    HAVE = True
except Exception:
    HAVE = False


def main():
    if not HAVE:
        print("cuStateVec not available, using numpy fallback")
        state = np.random.randn(2**10) + 1j * np.random.randn(2**10)
        t0 = time.time()
        state = state / np.linalg.norm(state)
        t1 = time.time()
        print("time_ms", (t1 - t0) * 1000)
        return

    handle = custatevec.create()
    n_qubits = 10
    dim = 1 << n_qubits
    state = np.zeros(dim, dtype=np.complex64)
    state[0] = 1.0

    t0 = time.time()
    custatevec.initialize_state_vector(handle, state)
    t1 = time.time()
    print("time_ms", (t1 - t0) * 1000)
    custatevec.destroy(handle)


if __name__ == "__main__":
    main()
