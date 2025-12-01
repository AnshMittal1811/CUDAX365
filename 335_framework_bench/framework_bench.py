
import time
import csv

def bench_torch():
    import torch
    import torch.nn as nn
    model = nn.Sequential(nn.Linear(128, 256), nn.ReLU(), nn.Linear(256, 10))
    x = torch.randn(256, 128)
    model.eval()
    t0 = time.perf_counter()
    for _ in range(100):
        model(x)
    t1 = time.perf_counter()
    return (t1 - t0) * 1000 / 100

def main():
    rows = []
    try:
        ms = bench_torch()
        rows.append(['torch_eager', ms])
    except Exception as e:
        rows.append(['torch_eager', f'error: {e}'])

    with open('framework_bench.csv', 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['framework', 'avg_ms'])
        writer.writerows(rows)
    print('Wrote framework_bench.csv')

if __name__ == '__main__':
    main()
