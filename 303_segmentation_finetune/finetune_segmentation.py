import numpy as np


def dice(pred, target):
    inter = (pred & target).sum()
    denom = pred.sum() + target.sum()
    return 2.0 * inter / denom if denom > 0 else 1.0


def main():
    images = np.load("../301_satellite_dataset/sat_images.npy")
    masks = np.load("../301_satellite_dataset/sat_masks.npy")

    pred = (images[:, 0] > 128).astype(np.uint8)
    d = dice(pred, masks)

    with open("finetune_log.txt", "w", encoding="utf-8") as f:
        f.write(f"dice={d:.4f}\n")

    print("Wrote finetune_log.txt")


if __name__ == "__main__":
    main()
