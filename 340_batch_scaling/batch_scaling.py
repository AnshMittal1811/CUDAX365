
import csv
import time

def main():
    try:
        import torch
        import torch.nn as nn
    except Exception:
        print('Torch not available; skipping batch scaling.')
        return

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = nn.Sequential(nn.Linear(256, 256), nn.ReLU(), nn.Linear(256, 10)).to(device)
    model.eval()

    batches = [1, 2, 4, 8, 16]
    rows = []
    for b in batches:
        x = torch.randn(b, 256, device=device)
        if device.type == 'cuda':
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(50):
            model(x)
        if device.type == 'cuda':
            torch.cuda.synchronize()
        t1 = time.perf_counter()
        ms = (t1 - t0) * 1000 / 50
        throughput = b / (ms / 1000.0)
        rows.append([b, ms, throughput])
        print(f'batch={b} ms={ms:.3f} imgs/s={throughput:.2f}')

    with open('batch_scaling.csv', 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['batch', 'avg_ms', 'throughput_imgs_s'])
        writer.writerows(rows)

if __name__ == '__main__':
    main()
