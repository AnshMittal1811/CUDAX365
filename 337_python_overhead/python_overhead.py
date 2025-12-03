
import time
import numpy as np

def main():
    try:
        import torch
        torch_ok = True
    except Exception:
        torch_ok = False

    if not torch_ok:
        x = np.random.standard_normal((1024,)).astype(np.float32)
        t0 = time.perf_counter()
        for _ in range(10000):
            _ = x + 1.0
        t1 = time.perf_counter()
        print(f'NumPy tiny op avg us: {(t1 - t0) * 1e6 / 10000:.2f}')
        return

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    x = torch.randn(1024, device=device)
    if device.type == 'cuda':
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(10000):
        _ = x + 1.0
    if device.type == 'cuda':
        torch.cuda.synchronize()
    t1 = time.perf_counter()
    print(f'Small kernel avg us: {(t1 - t0) * 1e6 / 10000:.2f}')

    x_big = torch.randn(1024 * 1024, device=device)
    if device.type == 'cuda':
        torch.cuda.synchronize()
    t2 = time.perf_counter()
    _ = x_big + 1.0
    if device.type == 'cuda':
        torch.cuda.synchronize()
    t3 = time.perf_counter()
    print(f'One big kernel ms: {(t3 - t2) * 1000:.2f}')

if __name__ == '__main__':
    main()
