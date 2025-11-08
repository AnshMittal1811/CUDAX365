
import argparse
import csv
import time
import numpy as np

def main():
    parser = argparse.ArgumentParser(description='Simulate real-time decoder loop')
    parser.add_argument('--rounds', type=int, default=200)
    parser.add_argument('--syndrome-bits', type=int, default=16)
    parser.add_argument('--error-bits', type=int, default=8)
    args = parser.parse_args()

    rng = np.random.default_rng(0)
    try:
        import torch
        import torch.nn as nn
        torch_available = True
    except Exception:
        torch_available = False

    timings = []
    if torch_available:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = nn.Sequential(
            nn.Linear(args.syndrome_bits, 64),
            nn.ReLU(),
            nn.Linear(64, args.error_bits),
        ).to(device)
        model.eval()
        x = torch.from_numpy(rng.standard_normal((1, args.syndrome_bits)).astype(np.float32)).to(device)
        if device.type == 'cuda':
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            for _ in range(args.rounds):
                start.record()
                with torch.no_grad():
                    model(x)
                end.record()
                torch.cuda.synchronize()
                timings.append(start.elapsed_time(end))
        else:
            for _ in range(args.rounds):
                t0 = time.perf_counter()
                with torch.no_grad():
                    model(x)
                t1 = time.perf_counter()
                timings.append((t1 - t0) * 1000.0)
    else:
        for _ in range(args.rounds):
            t0 = time.perf_counter()
            x = rng.standard_normal((1, args.syndrome_bits)).astype(np.float32)
            _ = x.sum(axis=1)
            t1 = time.perf_counter()
            timings.append((t1 - t0) * 1000.0)

    avg_ms = sum(timings) / max(1, len(timings))
    print(f'Average per-round latency: {avg_ms:.4f} ms')
    with open('decoder_realtime_timings.csv', 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['round', 'latency_ms'])
        for i, t in enumerate(timings):
            writer.writerow([i, t])

if __name__ == '__main__':
    main()
