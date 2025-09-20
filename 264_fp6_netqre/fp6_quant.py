import numpy as np


def quantize_fp6(x):
    max_val = np.max(np.abs(x))
    scale = max_val / 31.0 if max_val != 0 else 1.0
    q = np.clip(np.round(x / scale), -32, 31)
    return q * scale


def main():
    rng = np.random.RandomState(0)
    weights = rng.randn(1024).astype(np.float32)
    fp6 = quantize_fp6(weights)

    with open("fp6_quant_log.txt", "w", encoding="utf-8") as f:
        f.write(f"mean={fp6.mean():.6f}\n")
        f.write(f"std={fp6.std():.6f}\n")

    np.save("fp6_weights.npy", fp6)
    print("Wrote fp6_weights.npy and fp6_quant_log.txt")


if __name__ == "__main__":
    main()
