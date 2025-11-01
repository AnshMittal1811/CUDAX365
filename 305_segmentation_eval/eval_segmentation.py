import numpy as np

try:
    import matplotlib.pyplot as plt
except Exception:
    plt = None


def main():
    images = np.load("../301_satellite_dataset/sat_images.npy")
    masks = np.load("../301_satellite_dataset/sat_masks.npy")
    pred = (images[:, 0] > 128).astype(np.uint8)

    dice = (pred == masks).mean()

    with open("segmentation_eval.txt", "w", encoding="utf-8") as f:
        f.write(f"dice={dice:.4f}\n")

    if plt is not None:
        img = images[0].transpose(1, 2, 0)
        plt.figure(figsize=(4, 4))
        plt.imshow(img)
        plt.imshow(masks[0], alpha=0.4, cmap="Reds")
        plt.axis("off")
        plt.savefig("overlay.png", dpi=120)
    print("Wrote segmentation_eval.txt")


if __name__ == "__main__":
    main()
