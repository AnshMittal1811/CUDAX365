import time
import numpy as np

try:
    from cuquantum import contract
    HAVE_CUTENSORNET = True
except Exception:
    HAVE_CUTENSORNET = False


def main():
    np.random.seed(0)
    n, c, h, w = 8, 3, 32, 32
    k = 6  # out channels
    x = np.random.randn(n, c, h, w).astype(np.float32)
    w1 = np.random.randn(k, c).astype(np.float32)  # 1x1 conv weights

    t0 = time.time()
    if HAVE_CUTENSORNET:
        y = contract("nchw,kc->nkhw", x, w1)
    else:
        y = np.einsum("nchw,kc->nkhw", x, w1)
    t1 = time.time()
    print("output shape", y.shape)
    print("backend", "cutensornet" if HAVE_CUTENSORNET else "numpy")
    print("elapsed_ms", (t1 - t0) * 1000.0)


if __name__ == "__main__":
    main()
