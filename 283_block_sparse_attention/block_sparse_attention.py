import numpy as np


def block_sparse_attention(q, k, v, block=4):
    n = q.shape[0]
    out = np.zeros_like(q)
    for i in range(0, n, block):
        for j in range(0, n, block):
            if (i // block + j // block) % 2 == 0:
                qi = q[i:i + block]
                kj = k[j:j + block]
                vj = v[j:j + block]
                scores = qi @ kj.T
                scores = scores - scores.max(axis=1, keepdims=True)
                weights = np.exp(scores)
                weights /= weights.sum(axis=1, keepdims=True)
                out[i:i + block] += weights @ vj
    return out


def main():
    rng = np.random.RandomState(0)
    q = rng.randn(16, 16).astype(np.float32)
    k = rng.randn(16, 16).astype(np.float32)
    v = rng.randn(16, 16).astype(np.float32)

    out = block_sparse_attention(q, k, v)
    np.save("block_sparse_out.npy", out)
    print("Wrote block_sparse_out.npy")


if __name__ == "__main__":
    main()
