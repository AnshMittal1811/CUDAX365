
import time
import numpy as np

def bench_numpy(n):
    a = np.random.standard_normal((n, n)).astype(np.float32)
    b = np.random.standard_normal((n, n)).astype(np.float32)
    t0 = time.perf_counter()
    _ = a @ b
    t1 = time.perf_counter()
    return (t1 - t0) * 1000

def bench_torch(n):
    import torch
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    a = torch.randn(n, n, device=device)
    b = torch.randn(n, n, device=device)
    if device.type == 'cuda':
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    _ = a @ b
    if device.type == 'cuda':
        torch.cuda.synchronize()
    t1 = time.perf_counter()
    return (t1 - t0) * 1000

def main():
    sizes = [64, 128, 256, 384]
    try:
        import torch  # noqa: F401
        torch_ok = True
    except Exception:
        torch_ok = False
    for n in sizes:
        cpu_ms = bench_numpy(n)
        gpu_ms = bench_torch(n) if torch_ok else float('inf')
        print(f'n={n} cpu_ms={cpu_ms:.2f} gpu_ms={gpu_ms:.2f}')

if __name__ == '__main__':
    main()
