import csv
import random


def main():
    rng = random.Random(0)
    with open("learning_curve.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "baseline", "curriculum"])
        for epoch in range(20):
            base = 100 + rng.uniform(-10, 10)
            cur = base + rng.uniform(0, 8)
            writer.writerow([epoch, base, cur])
    print("Wrote learning_curve.csv")


if __name__ == "__main__":
    main()
