
import argparse
import time
import math
import numpy as np

def softmax(x):
    x = x - x.max(axis=-1, keepdims=True)
    exp = np.exp(x)
    return exp / exp.sum(axis=-1, keepdims=True)

def full_attention(x):
    scale = 1.0 / math.sqrt(x.shape[-1])
    scores = x @ x.T * scale
    attn = softmax(scores)
    return attn @ x

def block_attention(x, block):
    scale = 1.0 / math.sqrt(x.shape[-1])
    outs = []
    for start in range(0, x.shape[0], block):
        q = x[start:start+block]
        scores = q @ x.T * scale
        attn = softmax(scores)
        outs.append(attn @ x)
    return np.vstack(outs)

def main():
    parser = argparse.ArgumentParser(description='Measure attention memory/time')
    parser.add_argument('--tokens', type=int, default=512)
    parser.add_argument('--dim', type=int, default=64)
    parser.add_argument('--block', type=int, default=128)
    args = parser.parse_args()

    rng = np.random.default_rng(1)
    x = rng.standard_normal((args.tokens, args.dim)).astype(np.float32)

    t0 = time.perf_counter()
    full_attention(x)
    t1 = time.perf_counter()
    full_ms = (t1 - t0) * 1000.0
    full_mem_mb = (args.tokens * args.tokens * 4) / (1024 * 1024)

    t2 = time.perf_counter()
    block_attention(x, args.block)
    t3 = time.perf_counter()
    block_ms = (t3 - t2) * 1000.0
    block_mem_mb = (args.block * args.tokens * 4) / (1024 * 1024)

    print(f'Full attention time: {full_ms:.2f} ms, est scores mem: {full_mem_mb:.2f} MB')
    print(f'Block attention time: {block_ms:.2f} ms, est scores mem: {block_mem_mb:.2f} MB')
    print('Extrapolate savings by reducing block size or streaming more K/V blocks.')

if __name__ == '__main__':
    main()
