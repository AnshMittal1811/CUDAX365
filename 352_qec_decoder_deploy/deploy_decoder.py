
import time
import numpy as np

def main():
    try:
        import torch
        import torch.nn as nn
    except Exception:
        print('Torch not available')
        return

    model = nn.Sequential(nn.Linear(16, 32), nn.ReLU(), nn.Linear(32, 8)).eval()
    x = torch.randn(256, 16)
    t0 = time.perf_counter()
    for _ in range(100):
        model(x)
    t1 = time.perf_counter()
    print(f'CPU latency ms: {(t1 - t0) * 1000 / 100:.3f}')

    if torch.cuda.is_available():
        device = torch.device('cuda')
        model = model.to(device)
        xg = x.to(device)
        torch.cuda.synchronize()
        t2 = time.perf_counter()
        for _ in range(100):
            model(xg)
        torch.cuda.synchronize()
        t3 = time.perf_counter()
        print(f'GPU latency ms: {(t3 - t2) * 1000 / 100:.3f}')

    print('TensorRT export can be added if available.')

if __name__ == '__main__':
    main()
