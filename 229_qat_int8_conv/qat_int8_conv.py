import numpy as np


def quantize_int8(x):
    max_val = np.max(np.abs(x))
    scale = max_val / 127.0 if max_val != 0 else 1.0
    q = np.clip(np.round(x / scale), -128, 127)
    return q * scale


def main():
    x = np.random.rand(16, 16).astype(np.float32)
    qx = quantize_int8(x)

    with open("qat_log.txt", "w", encoding="utf-8") as f:
        f.write("qat_int8_done\n")
        f.write(f"mean={qx.mean():.6f}\n")

    print("Wrote qat_log.txt")


if __name__ == "__main__":
    main()
