import numpy as np

try:
    import torch
except Exception:
    torch = None


def main():
    rng = np.random.RandomState(0)
    dense = rng.randn(64, 64).astype(np.float32)
    mask = rng.rand(64, 64) > 0.8
    sparse = dense * mask

    if torch is None:
        out = sparse @ dense
        backend = "numpy"
    else:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        a = torch.tensor(sparse, device=device)
        b = torch.tensor(dense, device=device)
        out = a @ b
        backend = "torch"
        if device == "cuda":
            torch.cuda.synchronize()

    with open("spmm_log.txt", "w", encoding="utf-8") as f:
        f.write(f"backend={backend}\n")
        f.write(f"mean={float(out.mean()):.6f}\n")

    print("Wrote spmm_log.txt")


if __name__ == "__main__":
    main()
