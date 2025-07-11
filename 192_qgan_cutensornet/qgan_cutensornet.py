import argparse
import time

import numpy as np


def run_numpy(samples):
    kernel = np.array([0.25, 0.5, 0.25], dtype=np.float32)
    padded = np.pad(samples, ((1, 1), (0, 0)), mode="edge")
    out = np.zeros_like(samples)
    for i in range(samples.shape[0]):
        window = padded[i:i + 3]
        out[i] = (window[:, 0] * kernel).sum()
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", default="../191_qgan_pennylane/qgan_samples.npy")
    parser.add_argument("--out", default="cutensornet_log.txt")
    args = parser.parse_args()

    samples = np.load(args.samples)

    start = time.time()
    try:
        import cutensornet  # type: ignore
        backend = "cutensornet"
        out = run_numpy(samples)
    except Exception:
        backend = "numpy"
        out = run_numpy(samples)
    elapsed = time.time() - start

    np.save("conv_out.npy", out)
    with open(args.out, "w", encoding="utf-8") as f:
        f.write(f"backend={backend}\n")
        f.write(f"elapsed_sec={elapsed:.4f}\n")

    print(f"Wrote {args.out} and conv_out.npy")


if __name__ == "__main__":
    main()
