
import csv
import time
import numpy as np

def step_no_cache(x):
    scores = x @ x.T
    exp = np.exp(scores - scores.max(axis=-1, keepdims=True))
    attn = exp / exp.sum(axis=-1, keepdims=True)
    return attn @ x

def step_with_cache(q, k_cache, v_cache):
    scores = q @ k_cache.T
    exp = np.exp(scores - scores.max(axis=-1, keepdims=True))
    attn = exp / exp.sum(axis=-1, keepdims=True)
    return attn @ v_cache

def main():
    steps = 64
    dim = 64
    cache = np.random.standard_normal((steps, dim)).astype(np.float32)
    rows = []

    t0 = time.perf_counter()
    x = cache.copy()
    for _ in range(steps):
        step_no_cache(x)
    t1 = time.perf_counter()
    rows.append(['no_cache', (t1 - t0) * 1000])

    t2 = time.perf_counter()
    for i in range(steps):
        q = cache[i:i+1]
        step_with_cache(q, cache, cache)
    t3 = time.perf_counter()
    rows.append(['kv_cache', (t3 - t2) * 1000])

    with open('kv_cache_results.csv', 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['mode', 'total_ms'])
        writer.writerows(rows)

    for mode, ms in rows:
        print(f'{mode}: {ms:.2f} ms')

if __name__ == '__main__':
    main()
