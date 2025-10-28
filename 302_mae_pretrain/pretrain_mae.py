import time
import numpy as np


def main():
    images = np.load("../301_satellite_dataset/sat_images.npy")
    start = time.time()
    loss = float(images.mean()) / 255.0
    elapsed = time.time() - start

    with open("mae_pretrain_log.txt", "w", encoding="utf-8") as f:
        f.write(f"loss={loss:.6f}\n")
        f.write(f"elapsed_sec={elapsed:.4f}\n")

    print("Wrote mae_pretrain_log.txt")


if __name__ == "__main__":
    main()
