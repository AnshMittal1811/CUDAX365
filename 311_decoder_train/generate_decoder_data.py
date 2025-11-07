
import argparse
import numpy as np
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description='Generate simulated syndrome->error data')
    parser.add_argument('--samples', type=int, default=4096)
    parser.add_argument('--syndrome-bits', type=int, default=16)
    parser.add_argument('--error-bits', type=int, default=8)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--out', type=str, default='decoder_data.npz')
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    synd = rng.integers(0, 2, size=(args.samples, args.syndrome_bits)).astype(np.float32)
    weights = rng.standard_normal((args.syndrome_bits, args.error_bits)).astype(np.float32)
    logits = synd @ weights
    errors = (logits > 0).astype(np.float32)

    out_path = Path(args.out)
    np.savez(out_path, syndromes=synd, errors=errors, weights=weights)
    print(f'Wrote {out_path} with {args.samples} samples')

if __name__ == '__main__':
    main()
