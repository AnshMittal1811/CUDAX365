import numpy as np


def main():
    images = np.load("../301_satellite_dataset/sat_images.npy")
    masks = np.load("../301_satellite_dataset/sat_masks.npy")

    sar = images[:, 0:1].astype(np.float32) / 255.0
    fused = np.concatenate([images.astype(np.float32) / 255.0, sar], axis=1)

    pred = (fused[:, 0] > 0.5).astype(np.uint8)
    dice = (pred == masks).mean()

    with open("sar_fusion_log.txt", "w", encoding="utf-8") as f:
        f.write(f"dice={dice:.4f}\n")

    print("Wrote sar_fusion_log.txt")


if __name__ == "__main__":
    main()
