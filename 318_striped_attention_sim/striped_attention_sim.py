
import argparse
import time
import math
import numpy as np

def softmax(x):
    x = x - x.max(axis=-1, keepdims=True)
    exp = np.exp(x)
    return exp / exp.sum(axis=-1, keepdims=True)

def stripe_attention(x, stripes):
    scale = 1.0 / math.sqrt(x.shape[-1])
    outs = np.zeros_like(x)
    for s in range(stripes):
        idx = np.arange(s, x.shape[0], stripes)
        xs = x[idx]
        scores = xs @ xs.T * scale
        attn = softmax(scores)
        outs[idx] = attn @ xs
    return outs

def full_attention(x):
    scale = 1.0 / math.sqrt(x.shape[-1])
    scores = x @ x.T * scale
    attn = softmax(scores)
    return attn @ x

def main():
    parser = argparse.ArgumentParser(description='Simulate striped attention')
    parser.add_argument('--tokens', type=int, default=512)
    parser.add_argument('--dim', type=int, default=64)
    parser.add_argument('--stripes', type=int, default=2)
    args = parser.parse_args()

    rng = np.random.default_rng(2)
    x = rng.standard_normal((args.tokens, args.dim)).astype(np.float32)

    t0 = time.perf_counter()
    full_attention(x)
    t1 = time.perf_counter()

    t2 = time.perf_counter()
    stripe_attention(x, args.stripes)
    t3 = time.perf_counter()

    print(f'Full attention time: {(t1 - t0)*1000:.2f} ms')
    print(f'Striped attention time: {(t3 - t2)*1000:.2f} ms')
    print(f'Compute per stripe: {args.tokens // args.stripes} tokens')

if __name__ == '__main__':
    main()
