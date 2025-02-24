import cupy as cp


def main():
    state = cp.random.randn(16).astype(cp.float32)
    mat = state.reshape(4, 4)
    u, s, v = cp.linalg.svd(mat)
    print("singular values", s.get())


if __name__ == "__main__":
    main()
