import argparse
import time

import numpy as np


def atrous_filter(img, passes=2, step=2):
    kernel = np.array([1, 4, 6, 4, 1], dtype=np.float32)
    kernel2d = np.outer(kernel, kernel)
    kernel2d /= kernel2d.sum()

    out = img.copy()
    for _ in range(passes):
        padded = np.pad(out, 2 * step, mode="edge")
        result = np.zeros_like(out)
        for ky in range(-2, 3):
            for kx in range(-2, 3):
                weight = kernel2d[ky + 2, kx + 2]
                ys = (ky * step) + 2 * step
                xs = (kx * step) + 2 * step
                result += weight * padded[ys:ys + out.shape[0], xs:xs + out.shape[1]]
        out = result
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", default="../156_vulkan_atrous_denoise/noisy_image.npy")
    parser.add_argument("--out", default="denoise_timing.csv")
    parser.add_argument("--passes", type=int, default=2)
    args = parser.parse_args()

    img = np.load(args.image)

    t0 = time.time()
    _ = img.copy()
    t1 = time.time()

    t2 = time.time()
    denoised = atrous_filter(img, passes=args.passes)
    t3 = time.time()

    np.save("denoised_image.npy", denoised)

    no_denoise_ms = (t1 - t0) * 1000.0
    denoise_ms = (t3 - t2) * 1000.0

    with open(args.out, "w", encoding="utf-8") as f:
        f.write("mode,ms\n")
        f.write(f"noop,{no_denoise_ms:.4f}\n")
        f.write(f"denoise,{denoise_ms:.4f}\n")

    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
