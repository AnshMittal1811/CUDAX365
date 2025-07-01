import csv
import os

path = "../182_gl_nv_copy_image/copy_times.csv"

if not os.path.exists(path):
    raise SystemExit("Missing copy_times.csv; run Day 182 first")

values = {}
with open(path, "r", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    for row in reader:
        values[row["mode"]] = float(row["ms"])

ratio = values["staging"] / values["direct"] if values["direct"] else 0.0

with open("copy_compare.txt", "w", encoding="utf-8") as f:
    f.write(f"direct_ms={values['direct']:.4f}\n")
    f.write(f"staging_ms={values['staging']:.4f}\n")
    f.write(f"ratio={ratio:.3f}\n")

print("Wrote copy_compare.txt")
