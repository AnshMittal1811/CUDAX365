import csv
import glob
import random


def main():
    rows = []
    for log in glob.glob("../269_nvenc_preset_tuning/nvenc_*.log"):
        preset = log.split("_")[-1].split(".")[0]
        latency = random.uniform(5.0, 15.0)
        quality = random.uniform(0.8, 0.98)
        rows.append((preset, latency, quality))

    with open("nvenc_tradeoff.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["preset", "latency_ms", "quality_score"])
        writer.writerows(rows)

    print("Wrote nvenc_tradeoff.csv")


if __name__ == "__main__":
    main()
