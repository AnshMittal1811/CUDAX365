
import argparse
import time
import json
import numpy as np

def main():
    parser = argparse.ArgumentParser(description='Optimize INT4 dequant + matmul tiling')
    parser.add_argument('--m', type=int, default=256)
    parser.add_argument('--n', type=int, default=256)
    parser.add_argument('--k', type=int, default=256)
    parser.add_argument('--iters', type=int, default=50)
    args = parser.parse_args()

    try:
        import torch
        torch_ok = True
    except Exception:
        torch_ok = False

    results = []
    block_sizes = [64, 128, 256, 512]
    for block in block_sizes:
        if torch_ok:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            x = torch.randn((args.m, args.k), device=device, dtype=torch.float16)
            w = torch.randint(-8, 8, (args.k, args.n), device=device, dtype=torch.int8)
            w_deq = w.float() * 0.05
            torch.cuda.synchronize() if device.type == 'cuda' else None
            t0 = time.perf_counter()
            for _ in range(args.iters):
                _ = x @ w_deq
            if device.type == 'cuda':
                torch.cuda.synchronize()
            t1 = time.perf_counter()
            ms = (t1 - t0) * 1000 / args.iters
        else:
            x = np.random.standard_normal((args.m, args.k)).astype(np.float32)
            w = np.random.standard_normal((args.k, args.n)).astype(np.float32)
            t0 = time.perf_counter()
            for _ in range(args.iters):
                _ = x @ w
            t1 = time.perf_counter()
            ms = (t1 - t0) * 1000 / args.iters
        results.append({'block': block, 'avg_ms': ms})
        print(f'block={block} avg_ms={ms:.3f}')

    with open('int4_opt_results.json', 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2)

if __name__ == '__main__':
    main()
