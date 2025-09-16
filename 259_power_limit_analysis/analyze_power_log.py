import csv
import statistics


with open("../258_power_limit_tuning/power_limit_log.csv", "r", encoding="utf-8") as f:
    reader = csv.reader(f)
    header = next(reader, None)
    clocks = []
    for row in reader:
        if len(row) < 3:
            continue
        try:
            clocks.append(float(row[1].split()[0]))
        except Exception:
            continue

if clocks:
    avg = statistics.mean(clocks)
    stdev = statistics.pstdev(clocks)
else:
    avg = 0.0
    stdev = 0.0

with open("power_analysis.txt", "w", encoding="utf-8") as f:
    f.write(f"avg_clock={avg:.2f}\n")
    f.write(f"stdev_clock={stdev:.2f}\n")

print("Wrote power_analysis.txt")
