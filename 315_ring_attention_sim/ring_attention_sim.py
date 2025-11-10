
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

def ring_block_attention(x, block):
    scale = 1.0 / math.sqrt(x.shape[-1])
    outs = []
    for start in range(0, x.shape[0], block):
        q = x[start:start+block]
        scores = q @ x.T * scale
        attn = softmax(scores)
        outs.append(attn @ x)
    return np.vstack(outs)

def main():
    parser = argparse.ArgumentParser(description='Simulate ring-style block attention')
    parser.add_argument('--tokens', type=int, default=512)
    parser.add_argument('--dim', type=int, default=64)
    parser.add_argument('--block', type=int, default=128)
    args = parser.parse_args()

    rng = np.random.default_rng(0)
    x = rng.standard_normal((args.tokens, args.dim)).astype(np.float32)

    t0 = time.perf_counter()
    full_attention(x)
    t1 = time.perf_counter()

    t2 = time.perf_counter()
    ring_block_attention(x, args.block)
    t3 = time.perf_counter()

    print(f'Full attention time: {(t1 - t0)*1000:.2f} ms')
    print(f'Block attention time: {(t3 - t2)*1000:.2f} ms')

if __name__ == '__main__':
    main()
