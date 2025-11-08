
import argparse
import json
import os
import time
import numpy as np

def load_numpy_model(path):
    data = np.load(path)
    return {k: data[k] for k in data.files}

def numpy_forward(model, x):
    h = np.maximum(0, x @ model['w1'] + model['b1'])
    return h @ model['w2'] + model['b2']

def main():
    parser = argparse.ArgumentParser(description='Run decoder inference on CPU/GPU')
    parser.add_argument('--batch', type=int, default=1024)
    parser.add_argument('--syndrome-bits', type=int, default=16)
    parser.add_argument('--error-bits', type=int, default=8)
    parser.add_argument('--iters', type=int, default=200)
    args = parser.parse_args()

    rng = np.random.default_rng(0)
    x_np = rng.standard_normal((args.batch, args.syndrome_bits)).astype(np.float32)

    results = {}
    if os.path.exists('decoder_model.npz'):
        model_np = load_numpy_model('decoder_model.npz')
        t0 = time.perf_counter()
        for _ in range(args.iters):
            numpy_forward(model_np, x_np)
        t1 = time.perf_counter()
        results['cpu_numpy_ms'] = (t1 - t0) * 1000.0 / args.iters

    try:
        import torch
        import torch.nn as nn
        torch_available = True
    except Exception:
        torch_available = False

    if torch_available:
        device_cpu = torch.device('cpu')
        model = nn.Sequential(
            nn.Linear(args.syndrome_bits, 64),
            nn.ReLU(),
            nn.Linear(64, args.error_bits),
        )
        if os.path.exists('decoder_model.pt'):
            ckpt = torch.load('decoder_model.pt', map_location='cpu')
            model.load_state_dict(ckpt['state_dict'])
        model.eval()
        x_cpu = torch.from_numpy(x_np)
        with torch.no_grad():
            t0 = time.perf_counter()
            for _ in range(args.iters):
                model(x_cpu)
            t1 = time.perf_counter()
        results['cpu_torch_ms'] = (t1 - t0) * 1000.0 / args.iters

        if torch.cuda.is_available():
            device = torch.device('cuda')
            model = model.to(device)
            x_gpu = x_cpu.to(device)
            torch.cuda.synchronize()
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            with torch.no_grad():
                for _ in range(args.iters):
                    model(x_gpu)
            end.record()
            torch.cuda.synchronize()
            results['gpu_ms'] = start.elapsed_time(end) / args.iters

    with open('decoder_infer_results.json', 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2)
    print(json.dumps(results, indent=2))

if __name__ == '__main__':
    main()
