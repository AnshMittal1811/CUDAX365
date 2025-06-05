import numpy as np


def atrous_filter(img, step=2, passes=1):
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


if __name__ == "__main__":
    img = np.load("noisy_image.npy")
    denoised = atrous_filter(img, step=2, passes=2)
    np.save("denoised_image.npy", denoised)
    print("Wrote denoised_image.npy")
