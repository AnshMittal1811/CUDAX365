import argparse
import numpy as np


def psnr(a, b):
    mse = np.mean((a - b) ** 2)
    if mse == 0.0:
        return 99.0
    return 20.0 * np.log10(1.0 / np.sqrt(mse))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--clean", default="../156_vulkan_atrous_denoise/noisy_image.npy")
    parser.add_argument("--denoised", default="../158_denoise_timing/denoised_image.npy")
    parser.add_argument("--out", default="quality_metrics.txt")
    args = parser.parse_args()

    clean = np.load(args.clean)
    denoised = np.load(args.denoised)

    score = psnr(clean, denoised)
    with open(args.out, "w", encoding="utf-8") as f:
        f.write(f"psnr={score:.3f}\n")

    print(f"Wrote {args.out} (psnr={score:.3f})")


if __name__ == "__main__":
    main()
