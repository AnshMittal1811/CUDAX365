import csv
import random


def main():
    rng = random.Random(0)
    rows = []
    for step in range(200):
        fp16 = 0.5 + rng.uniform(-0.05, 0.05)
        fp8 = 0.5 + rng.uniform(-0.08, 0.08)
        rows.append((step, fp16, fp8))

    with open("fp8_stability.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["step", "fp16_loss", "fp8_loss"])
        writer.writerows(rows)

    print("Wrote fp8_stability.csv")


if __name__ == "__main__":
    main()
