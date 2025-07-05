import argparse
import numpy as np

try:
    import torch
    import torch.nn as nn
except Exception:
    torch = None
    nn = None


def quantize_fp4(tensor):
    max_val = tensor.abs().max()
    if max_val == 0:
        return tensor
    scale = max_val / 7.0
    q = (tensor / scale).round().clamp(-8, 7)
    return q * scale


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="fp4_quant_summary.txt")
    args = parser.parse_args()

    if torch is None:
        with open(args.out, "w", encoding="utf-8") as f:
            f.write("torch not available\n")
        print("torch not available; wrote summary")
        return

    model = nn.Sequential(
        nn.Conv2d(3, 8, 3, padding=1),
        nn.ReLU(),
        nn.Conv2d(8, 2, 1),
    )

    with torch.no_grad():
        for name, param in model.named_parameters():
            param.copy_(quantize_fp4(param))

    with open(args.out, "w", encoding="utf-8") as f:
        f.write("Applied simulated FP4 quantization to weights\n")

    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
