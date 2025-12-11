
import csv
import time

def main():
    try:
        import torch
        import torch.nn as nn
    except Exception:
        print('Torch not available')
        return

    model = nn.Sequential(nn.Linear(256, 256), nn.ReLU(), nn.Linear(256, 10))
    x = torch.randn(256, 256)

    results = []

    # FP32
    t0 = time.perf_counter()
    for _ in range(50):
        model(x)
    t1 = time.perf_counter()
    results.append(['fp32', (t1 - t0) * 1000 / 50])

    # FP16 (if CUDA)
    if torch.cuda.is_available():
        device = torch.device('cuda')
        model_fp16 = model.to(device).half()
        x_fp16 = x.to(device).half()
        torch.cuda.synchronize()
        t2 = time.perf_counter()
        for _ in range(50):
            model_fp16(x_fp16)
        torch.cuda.synchronize()
        t3 = time.perf_counter()
        results.append(['fp16', (t3 - t2) * 1000 / 50])
    else:
        results.append(['fp16', 'cuda_unavailable'])

    # INT8 (CPU dynamic quant)
    try:
        qmodel = torch.quantization.quantize_dynamic(model, {torch.nn.Linear}, dtype=torch.qint8)
        t4 = time.perf_counter()
        for _ in range(50):
            qmodel(x)
        t5 = time.perf_counter()
        results.append(['int8', (t5 - t4) * 1000 / 50])
    except Exception as e:
        results.append(['int8', f'error: {e}'])

    with open('precision_compare.csv', 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['precision', 'avg_ms'])
        writer.writerows(results)
    print(results)

if __name__ == '__main__':
    main()
