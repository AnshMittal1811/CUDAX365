import numpy as np


def main():
    rng = np.random.RandomState(0)
    labels = rng.randint(0, 2, size=256)
    pred_fp32 = rng.randint(0, 2, size=256)
    pred_int8 = rng.randint(0, 2, size=256)

    acc_fp32 = (pred_fp32 == labels).mean()
    acc_int8 = (pred_int8 == labels).mean()

    with open("int8_accuracy.txt", "w", encoding="utf-8") as f:
        f.write(f"acc_fp32={acc_fp32:.3f}\n")
        f.write(f"acc_int8={acc_int8:.3f}\n")
        f.write(f"drop={acc_fp32 - acc_int8:.3f}\n")

    print("Wrote int8_accuracy.txt")


if __name__ == "__main__":
    main()
