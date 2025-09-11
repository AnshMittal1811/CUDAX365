import time
import numpy as np

try:
    import torch
except Exception:
    torch = None


def train_numpy():
    rng = np.random.RandomState(0)
    data = rng.randn(1024, 64).astype(np.float32)
    targets = rng.randn(1024, 1).astype(np.float32)
    weights = np.zeros((64, 1), dtype=np.float32)
    lr = 0.01
    for _ in range(2):
        preds = data @ weights
        grad = (data.T @ (preds - targets)) / len(data)
        weights -= lr * grad
    return float(np.mean((data @ weights - targets) ** 2))


def train_torch():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    data = torch.randn(1024, 64, device=device)
    targets = torch.randn(1024, 1, device=device)
    weights = torch.zeros(64, 1, device=device, requires_grad=True)
    opt = torch.optim.SGD([weights], lr=0.01)
    for _ in range(2):
        opt.zero_grad()
        preds = data @ weights
        loss = torch.mean((preds - targets) ** 2)
        loss.backward()
        opt.step()
    if device == "cuda":
        torch.cuda.synchronize()
    return float(loss.item())


def main():
    start = time.time()
    try:
        loss = train_torch() if torch is not None else train_numpy()
        backend = "torch" if torch is not None else "numpy"
    except Exception:
        loss = train_numpy()
        backend = "numpy"
    elapsed = time.time() - start

    with open("rlhf_train_log.txt", "w", encoding="utf-8") as f:
        f.write(f"backend={backend}\n")
        f.write(f"loss={loss:.6f}\n")
        f.write(f"elapsed_sec={elapsed:.4f}\n")

    print("Wrote rlhf_train_log.txt")


if __name__ == "__main__":
    main()
