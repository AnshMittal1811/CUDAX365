
import multiprocessing as mp
import time

def worker(iters):
    try:
        import torch
        import torch.nn as nn
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = nn.Sequential(nn.Linear(128, 128), nn.ReLU()).to(device)
        x = torch.randn(256, 128, device=device)
        if device.type == 'cuda':
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(iters):
            model(x)
        if device.type == 'cuda':
            torch.cuda.synchronize()
        t1 = time.perf_counter()
        return t1 - t0
    except Exception:
        t0 = time.perf_counter()
        for _ in range(iters * 1000):
            pass
        t1 = time.perf_counter()
        return t1 - t0

def main():
    iters = 100
    with mp.Pool(processes=2) as pool:
        times = pool.map(worker, [iters, iters])
    print('Process times (s):', times)

if __name__ == '__main__':
    main()
