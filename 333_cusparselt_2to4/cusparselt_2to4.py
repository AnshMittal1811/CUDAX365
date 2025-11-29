
import numpy as np
import time

def prune_2to4(w):
    w = w.copy()
    for row in range(w.shape[0]):
        for col in range(0, w.shape[1], 4):
            block = w[row, col:col+4]
            idx = np.argsort(np.abs(block))[:2]
            block[idx] = 0.0
            w[row, col:col+4] = block
    return w

def main():
    m, k, n = 128, 128, 128
    a = np.random.standard_normal((m, k)).astype(np.float32)
    w = np.random.standard_normal((k, n)).astype(np.float32)

    t0 = time.perf_counter()
    dense = a @ w
    t1 = time.perf_counter()

    w_pruned = prune_2to4(w)
    t2 = time.perf_counter()
    sparse = a @ w_pruned
    t3 = time.perf_counter()

    print(f'Dense GEMM ms: {(t1 - t0)*1000:.2f}')
    print(f'Prune time ms: {(t2 - t1)*1000:.2f}')
    print(f'Pruned GEMM ms: {(t3 - t2)*1000:.2f}')
    print('cuSPARSELt not available in this environment; using NumPy fallback.')

if __name__ == '__main__':
    main()
