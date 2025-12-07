
import csv
import time
import numpy as np

def attention(x):
    scores = x @ x.T
    scores = scores - scores.max(axis=-1, keepdims=True)
    exp = np.exp(scores)
    attn = exp / exp.sum(axis=-1, keepdims=True)
    return attn @ x

def main():
    lengths = [128, 256, 512]
    rows = []
    for n in lengths:
        x = np.random.standard_normal((n, 64)).astype(np.float32)
        t0 = time.perf_counter()
        attention(x)
        t1 = time.perf_counter()
        ms = (t1 - t0) * 1000
        tokens_per_s = n / (ms / 1000.0)
        rows.append([n, ms, tokens_per_s])
        print(f'n={n} ms={ms:.2f} tokens/s={tokens_per_s:.2f}')

    with open('context_scaling.csv', 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['context', 'time_ms', 'tokens_per_s'])
        writer.writerows(rows)

if __name__ == '__main__':
    main()
