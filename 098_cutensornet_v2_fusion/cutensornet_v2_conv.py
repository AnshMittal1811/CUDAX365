import numpy as np

try:
    from cuquantum import contract
    HAVE = True
except Exception:
    HAVE = False


def main():
    n, c, h, w = 2, 3, 16, 16
    k = 4
    x = np.random.randn(n, c, h, w).astype(np.float32)
    w1 = np.random.randn(k, c).astype(np.float32)
    if HAVE:
        y = contract("nchw,kc->nkhw", x, w1)
        backend = "cutensornet"
    else:
        y = np.einsum("nchw,kc->nkhw", x, w1)
        backend = "numpy"
    print("backend", backend, "shape", y.shape)


if __name__ == "__main__":
    main()
